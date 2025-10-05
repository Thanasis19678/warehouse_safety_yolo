import cv2
import numpy as np
import time

# Homography Fuction
class ViewTransformer:
    # view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(
                reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


def iou_xyxy(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    denom = areaA + areaB - inter
    return inter/denom if denom>0 else 0.0


def compute_velocity_with_kalman1(tracker_dict, object_id, measurement, dt,
                                  sigma_meas,  # std της μέτρησης (px ή μέτρα)
                                  sigma_accel,  # std επιτάχυνσης (px/s^2 ή μ/s^2)
                                  nis_maneuver=6.0,  # κατώφλι NIS για "στροφή"
                                  q_boost=10.0,  # πόσο φουσκώνουμε το Q σε στροφή
                                  nis_outlier=21.0,  # gate: αγνόηση κακών μετρήσεων
                                  fade_P=1.02,  # fading memory (1.0 = off)
                                  ):
    """
    Kalman (CA: [x,y,vx,vy,ax,ay]) :
    - tracker_dict: dict[object_id]
    - measurement: [x, y]
    - dt: Δευτερόλεπτα
    Επιστρέφει: np.array([vx, vy], float32)
    """
    meas = np.asarray(measurement, dtype=np.float32).reshape(2, 1)

    entry = tracker_dict.get(object_id)
    dt=dt
    if entry is None:
        # --------- INIT: 6x2 Kalman (CA) ----------
        kf = cv2.KalmanFilter(6, 2)  # state: [x,y,vx,vy,ax,ay], meas: [x,y]

        # H
        H = np.zeros((2, 6), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        kf.measurementMatrix = H

        # F(dt)
        # dt0 =1/fps
        F = np.array([[1,0,dt,0, 0.5*dt*dt, 0],
                      [0,1,0, dt,0,        0.5*dt*dt],
                      [0,0,1, 0, dt,       0],
                      [0,0,0, 1, 0,        dt],
                      [0,0,0, 0, 1,        0],
                      [0,0,0, 0, 0,        1]], dtype=np.float32)
        kf.transitionMatrix = F

        # R (measurement noise)
        var_m = float(sigma_meas ** 2)
        R = np.diag([0.5*var_m, var_m]).astype(np.float32)
        kf.measurementNoiseCov = R

        # Q (process noise)
        var_a = float(sigma_accel**2)
        Q = np.diag([0, 0, 0, 0, var_a, var_a]).astype(np.float32)
        Q[2, 4] = Q[4, 2] = 0.1 * var_a
        Q[3, 5] = Q[5, 3] = 0.1 * var_a
        kf.processNoiseCov = Q

        # P (αβεβαιότητα κατάστασης): μικρή στη θέση, μεγάλη σε ταχύτητα/επιτάχ.
        P = np.diag([10.0, 10.0, 1e3, 1e3, 1e3, 1e3]).astype(np.float32)
        kf.errorCovPost = P


        # αρχική κατάσταση (post)
        kf.statePost = np.zeros((6, 1), dtype=np.float32)
        kf.statePost[0:2, 0] = meas[:, 0]  # x,y

        tracker_dict[object_id] = {
            "kf": kf,
            "last_meas": meas.copy(),
            "seen": 1,
            "BASE_Q": Q.copy(),
            "BASE_R": R.copy(),
            # "last_time": now
        }
        return np.array([0.0, 0.0], dtype=np.float32)

    # --------- UPDATE EXISTING FILTER ----------
    kf = entry["kf"]



    # F(dt) update
    F = kf.transitionMatrix
    F[0, 2] = dt; F[1, 3] = dt
    F[0, 4] = 0.5 * dt*dt; F[1, 5] = 0.5 * dt*dt
    F[2, 4] = dt; F[3, 5] = dt
    kf.transitionMatrix = F

    #  # x_k   ← x + vx·dt
    # F[1, 3] = dt         # y_k   ← y + vy·dt
    # F[0, 4] = 0.5*dt^2   # x_k   ← + 0.5 ax·dt^2
    # F[1, 5] = 0.5*dt^2   # y_k   ← + 0.5 ay·dt^2
    # F[2, 4] = dt         # vx_k  ← vx + ax·dt
    # F[3, 5] = dt         # vy_k  ← vy + ay·dt

    # Reset base Q/R
    BASE_Q = entry["BASE_Q"]
    BASE_R = entry["BASE_R"]
    kf.processNoiseCov = BASE_Q.copy()
    kf.measurementNoiseCov = BASE_R.copy()

    # Predict
    kf.predict()

    # Warm-start ταχύτητας στο 2ο frame
    if entry["seen"] == 1 and dt > 1e-6:
        dv = (meas - entry["last_meas"]) / dt   # 2x1
        kf.statePre[2, 0] = float(dv[0, 0])  # vx
        kf.statePre[3, 0] = float(dv[1, 0])  # vy

    # --- Innovation / NIS για ανίχνευση μανούβρας ---
    H = kf.measurementMatrix
    z_pred = H @ kf.statePre
    innov  = meas - z_pred                       # 2x1
    S      = H @ kf.errorCovPre @ H.T + kf.measurementNoiseCov  # 2x2

    # προστασία αντιστροφής 2x2
    try:
        S_inv = np.linalg.inv(S.astype(np.float64))
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S.astype(np.float64))

    nis = float(innov.T @ S_inv @ innov)

    # --- Αν μανούβρα: φούσκωσε προσωρινά το Q ---
    if nis > nis_maneuver:
        kf.processNoiseCov *= float(q_boost)

    # --- Fading memory: γρηγορότερη προσαρμογή ---
    if fade_P > 1.0:
        kf.errorCovPre  *= float(fade_P)

    # --- Outlier gating ---
    if nis <= nis_outlier:
        kf.correct(meas)
    else:
        # αγνόησε τη μέτρηση
        pass

    # update meta
    entry["last_meas"] = meas.copy()
    entry["seen"] += 1

    vx = float(kf.statePost[2, 0])
    vy = float(kf.statePost[3, 0])
    return np.array([vx, vy], dtype=np.float32)



def compute_velocity_with_kalman(tracker_dict, object_id, measurement, dt):
    """
    Εκτιμά [vx, vy] με Kalman (Constant Velocity).
    - tracker_dict: dict[object_id]
    - measurement: [x, y]
    - dt: Δευτερόλεπτα
    """
    meas = np.asarray(measurement, dtype=np.float32).reshape(2, 1)

    entry = tracker_dict.get(object_id, None)
    if entry is None:
        # --- init ---
        kf = cv2.KalmanFilter(4, 2)
        # State: [x, y, vx, vy]
        kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                        [0, 1, 0, dt],
                                        [0, 0, 1,  0],
                                        [0, 0, 0,  1]], dtype=np.float32)
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], dtype=np.float32)

        # Αρχικές καταστάσεις
        kf.statePost = np.zeros((4, 1), dtype=np.float32)
        kf.statePost[0:2, 0] = meas[:, 0]

        # Αρχικό P: μεγάλη αβεβαιότητα στη ταχύτητα
        kf.errorCovPost = np.diag([10.0, 10.0, 1e3, 1e3]).astype(np.float32)

        # Αρχικά R (std~2px -> var=4). Μπορείς να τα κουρδίσεις
        kf.measurementNoiseCov = np.diag([4.0, 4.0]).astype(np.float32)

        # Αρχικό Q με το τρέχον dt
        dt2, dt3, dt4 = dt*dt, dt*dt*dt, (dt*dt)*(dt*dt)
        q = 5.0  # ένταση τυχ. επιτάχυνσης (ρύθμισέ το)
        Q = np.array([[dt4/4,   0,      dt3/2,  0     ],
                      [0,       dt4/4,  0,      dt3/2 ],
                      [dt3/2,   0,      dt2,    0     ],
                      [0,       dt3/2,  0,      dt2   ]], dtype=np.float32) * q
        kf.processNoiseCov = Q

        tracker_dict[object_id] = {
            "kf": kf,
            "last_meas": meas.copy(),
            "seen": 1
        }
        # 1η κλήση: δεν έχουμε ακόμα καλή vx,vy
        return np.array([0.0, 0.0], dtype=np.float32)

    # --- use existing filter ---
    kf = entry["kf"]

    # Ενημέρωσε F με νέο dt
    F = kf.transitionMatrix
    F[0, 2] = dt
    F[1, 3] = dt
    kf.transitionMatrix = F

    # Ενημέρωσε Q(dt)
    dt2, dt3, dt4 = dt*dt, dt*dt*dt, (dt*dt)*(dt*dt)
    q = 5.0
    Q = np.array([[dt4/4,   0,      dt3/2,  0     ],
                  [0,       dt4/4,  0,      dt3/2 ],
                  [dt3/2,   0,      dt2,    0     ],
                  [0,       dt3/2,  0,      dt2   ]], dtype=np.float32) * q
    kf.processNoiseCov = Q

    # Predict -> ενημερώνει statePre
    kf.predict()

    # Warm-start ταχύτητας στο 2ο frame
    if entry["seen"] == 1 and dt > 1e-6:
        dv = (meas - entry["last_meas"]) / dt   # 2x1
        kf.statePre[2:, 0] = dv[:, 0]

    # Correct
    kf.correct(meas)

    # Update meta
    entry["last_meas"] = meas.copy()
    entry["seen"] += 1

    vx = float(kf.statePost[2, 0])
    vy = float(kf.statePost[3, 0])
    return np.array([vx, vy], dtype=np.float32)


# Fuction : Compute ttce, dce
def compute_ttce_dce(pos1, vel1, pos2, vel2, eps=1e-9):
    p1 = np.asarray(pos1, dtype=float)
    v1 = np.asarray(vel1, dtype=float)
    p2 = np.asarray(pos2, dtype=float)
    v2 = np.asarray(vel2, dtype=float)

    r = p1 - p2
    v = v1 - v2

    # Αν έχει NaN/Inf οτιδήποτε -> επέστρεψε ασφαλώς
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v))):
        return np.inf, float(np.linalg.norm(r))

    vv = float(np.dot(v, v))
    if vv <= eps:
        # Μηδενική/αμελητέα σχετική ταχύτητα -> η απόσταση δεν αλλάζει ουσιαστικά
        return np.inf, float(np.linalg.norm(r))

    ttce = - float(np.dot(r, v)) / vv
    if not np.isfinite(ttce):
        return np.inf, float(np.linalg.norm(r))

    # Κοιτάμε μόνο μέλλον (όχι αρνητικούς χρόνους)
    if ttce < 0.0:
        ttce = 0.0

    dce = float(np.linalg.norm(r + ttce * v))
    return ttce, dce

# Compute TTC
def ttc1(p1, v1, p2, v2, d=1.0, eps=1e-9):
    """
    Time-To-Collision με σταθερές ταχύτητες.
    p1, v1, p2, v2: array-like θέσεις/ταχύτητες (px ή μέτρα, ίδια μονάδα)
    d: κατώφλι απόστασης για "σύγκρουση" (π.χ. 1.0)
    eps: μικρός αριθμός για αριθμητική σταθερότητα
    """
    p1 = np.asarray(p1, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    r = p1 - p2         # σχετική θέση
    v = v1 - v2         # σχετική ταχύτητα

    A = np.dot(v, v)
    B = np.dot(r, v)
    C = np.dot(r, r) - d*d

    # ήδη εντός της απόστασης d
    if C <= 0.0:
        return 0.0

    # μηδενική/αμελητέα σχετική ταχύτητα -> δεν θα πλησιάσουν
    if A <= eps:
        return np.inf

    # διακρίνουσα
    disc = B*B - A*C
    if disc < 0.0:
        return np.inf

    # πρώτη ρίζα (είσοδος στο δίσκο ακτίνας d)
    t_enter = (-B - np.sqrt(disc)) / A
    return t_enter if t_enter >= 0.0 else np.inf


def compute_W(rel_pos_f, Va, Fbf, weights, eps=1e-6):
    # null checks
    if rel_pos_f is None or Va is None or Fbf is None:
        return None

    rel_pos_f = np.asarray(rel_pos_f, dtype=float)
    Va        = np.asarray(Va,        dtype=float)
    Fbf       = np.asarray(Fbf,       dtype=float)

    # finite checks
    if not (np.all(np.isfinite(rel_pos_f)) and
            np.all(np.isfinite(Va)) and
            np.all(np.isfinite(Fbf))):
        return None

    nr = np.linalg.norm(rel_pos_f)
    nv = np.linalg.norm(Va)
    nf = np.linalg.norm(Fbf)
    if (nr < eps) or (nv < eps) or (nf < eps):
        return None

    rel_pos_p = -rel_pos_f

    # cosines (με guards για NaN)
    cosf1 = float(np.dot(Va,  rel_pos_p) / (nv * np.linalg.norm(rel_pos_p)))
    cosf2 = float(np.dot(Fbf, rel_pos_f) / (nf * nr))
    if not (np.isfinite(cosf1) and np.isfinite(cosf2)):
        return None

    green, yellow, orange, red = weights

    # w1 (εργάτης)
    if 0.9 <= cosf1 <= 1.0:
        w1 = green
    elif 0.0 < cosf1 < 0.9:
        w1 = yellow
    elif -0.707 <= cosf1 < 0.0:
        w1 = orange
    elif -1.0 <= cosf1 < -0.707:
        w1 = red
    else:
        w1 = green  # default bucket

    # w2 (forklift)
    if 0.9 <= cosf2 <= 1.0:
        w2 = green
    elif 0.0 < cosf2 < 0.9:
        w2 = yellow
    elif -0.707 <= cosf2 < 0.0:
        w2 = orange
    elif -1.0 <= cosf2 < -0.707:
        w2 = red
    else:
        w2 = green

    return w1, w2


def risk_score(dist, ttc, ttce, dce, W1, W2,
               min_d=4,
               d=1.0,
               delta_d=0.1, delta_t=0.1,
               Tmax=6.0,
               D0=2.0, T0=2.0,
               alpha=5, beta=8, gamma=2,
               clip_out=True):
    """
    dist: τρέχουσα απόσταση (ίδιες μονάδες με d)
    ttc: time-to-threshold d (s). Αν <=0 ή inf → αγνοείται.
    ttce: time-to-closest-encounter (s). Αν <=0 ή inf → αγνοείται.
    dce: distance at closest encounter (ίδιες μονάδες με d)
    W: συντελεστής κατεύθυνσης/σχετικής διάταξης (π.χ. w1*w2)
    """
    # f_dist
    f_dist = 1.0 / max(dist, delta_d)

    # f_ttc
    if np.isfinite(ttc) and (ttc > 0.0):
        f_ttc = 1.0 / (min(ttc, Tmax) + delta_t)
    else:
        f_ttc = 0.0

    # f_near
    if np.isfinite(ttce) and (ttce > 0.0) and np.isfinite(dce):
        f_near = np.exp(-(max(dce - d, 0.0)) / D0) * np.exp(-(ttce) / T0)
    else:
        f_near = 0.0

    if dist > min_d :
        W1 = 0.5*W1
        W2 = 0.5*W2

    risk = float(W1) *float(W2)* (alpha * f_dist + beta * f_ttc + gamma * f_near)

    # if clip_out:
    #     # προαιρετική συμπίεση σε 0..100
    #     risk = 100.0 * (1.0 - np.exp(-risk))
    return risk
