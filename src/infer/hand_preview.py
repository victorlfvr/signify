# signify/src/hand_preview.py
import argparse, time, cv2, mediapipe as mp

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)   # 640x480 = fluide
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--det", type=float, default=0.6)   # seuils un peu + hauts
    ap.add_argument("--track", type=float, default=0.6)
    return ap.parse_args()

def open_cam(idx, w, h):
    # Sur Windows, MSMF est souvent plus stable que DSHOW
    cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # évite l'accumulation de frames
    # MJPG réduit la charge CPU si supporté
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap

def main():
    args = parse_args()
    cap = open_cam(args.cam, args.width, args.height)

    mp_draw = mp.solutions.drawing_utils
    mp_conn = mp.solutions.hands.HAND_CONNECTIONS

    # Utiliser le context manager évite des fuites/locks
    with mp.solutions.hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=args.det,
        min_tracking_confidence=args.track,
    ) as hands:

        ema_fps = 0.0
        last = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                # caméra qui décroche → on tente une réouverture rapide
                cap.release()
                cap = open_cam(args.cam, args.width, args.height)
                continue

            # MediaPipe travaille plus vite si l'image est non modifiable
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            res = hands.process(img)

            frame.flags.writeable = True
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, lm, mp_conn)

            # FPS lissé (EMA) pour un affichage stable
            now = time.time()
            inst = 1.0 / max(now - last, 1e-6)
            last = now
            ema_fps = 0.9 * ema_fps + 0.1 * inst if ema_fps > 0 else inst
            cv2.putText(frame, f"FPS {ema_fps:5.1f}", (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Hands Preview", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break
            elif k == ord('1'):
                args.cam = 0; cap.release(); cap = open_cam(0, args.width, args.height)
            elif k == ord('2'):
                args.cam = 1; cap.release(); cap = open_cam(1, args.width, args.height)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()