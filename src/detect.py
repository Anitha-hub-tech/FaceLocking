# src/detect.py

import cv2


def main():

    # Haar cascade
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face = cv2.CascadeClassifier(cascade_path)

    if face.empty():
        raise RuntimeError(
            f"Failed to load cascade: {cascade_path}"
        )

    # External camera = index 1
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "External camera not opened. Try camera index 0/1/2."
        )

    # Use a reasonable resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Haar face detect using external camera (index 1).")
    print("Press 'q' to quit.")

    while True:

        ok, frame = cap.read()

        if not ok:
            print("Failed to read frame.")
            break

        # Grayscale for Haar
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Face detection
        faces = face.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        # Draw faces
        for (x, y, w, h) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

        cv2.imshow(
            "External Camera - Face Detection",
            frame
        )

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
