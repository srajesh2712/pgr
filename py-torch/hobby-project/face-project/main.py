import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(
    frame,
    actions=['emotion', 'age'],
    enforce_detection=False,
    detector_backend='skip'
)
        emotion = result[0]['dominant_emotion']
        age = result[0]['age']

        # Draw HUD
        cv2.putText(frame, f"Emotion: {emotion}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Age: {age}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        cv2.putText(frame, f"Err: {str(e)[:40]}", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Face Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()