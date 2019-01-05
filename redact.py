import argparse
import logging

import cv2
import numpy as np


class Redactor(object):

    detector = None


    def __init__(self, xml_classifier_path: str):
        self.detector = self._load_detector(xml_classifier_path)


    @classmethod
    def fromXmlClassifier(cls, xml_classifier_path: str):
        return cls(xml_classifier_path)


    def _load_detector(self, xml_classifier_path: str):
        """
        Load the face detector.
        
        Parameters:
        classifier_path: OpenCV haar cascade xml file
    
        Returns:
        detector: OpenCV haar face detector
        """
        face_detector = cv2.CascadeClassifier()
        face_detector.load(xml_classifier_path)
        return face_detector
    
    
    def blur_faces(self, face_detections: np.ndarray, frame: np.ndarray):
        """
        Blur faces in the frame.
    
        Parameters:
        face_detections: locations of faces from the detector
        frame: OpenCV image
    
        Returns:
        OpenCV image: faces blurred
        """
        if len(face_detections):
            for f in face_detections:
                x, y, w, h = [ v for v in f ]
                face = frame[y:y+h, x:x+w]
                face = cv2.GaussianBlur(face,(23, 23), 30)
                frame[y:y+h, x:x+w] = face
        return frame
    
    
    def detect_faces(self, frame: np.ndarray):
        """
        Detect faces in the frame.
    
        Parameters:
        detector: OpenCV haar face detector
        frame: OpenCV image
    
        Returns:
        faces: face detections
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2, flags=0, minSize=(30,30))
        return faces
    
    
    def redact_frame(self, frame: np.ndarray):
        """
        Perform face redaction in the frame.
    
        Face redaction includes face detection and blurring in those regions.
    
        Parameters:
        detector: OpenCV haar face detector
        frame: OpenCV image
    
        Returns:
        redacted_image: OpenCV image
        """
        faces = self.detect_faces(frame)
        redacted_image = self.blur_faces(faces, frame)
        return redacted_image


def process_webcam(webcam_num: int, redactor: Redactor):
    """
    Process a web camera in live mode.

    Parameters:
    webcam_num: OpenCV webcam number (typically 0 or 1)
    detector: OpenCV harr face detector

    Returns:
    nothing
    """
    cap = cv2.VideoCapture(webcam_num)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            redacted_image = redactor.redact_frame(frame)
            cv2.imshow('redact', redacted_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_video(video_path: str, redactor: Redactor):
    """
    Process a video file.

    Parameters:
    video_path: path to video file
    detector: OpenCV haar face detector

    Returns:
    nothing
    """
    cap = cv2.VideoCapture(video_path)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            redacted_image = redactor.redact_frame(frame)
            cv2.imshow('redact', redacted_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def process_image(image_path: str, redactor: Redactor):
    """
    Process a single image file

    Parameters:
    image_path: path to image file
    detector: OpenCV haar face detector
   
    Returns:
    nothing
    """
    image = cv2.imread(image_path)
    redacted_image = redactor.redact_frame(image)
    cv2.imshow('redact',redacted_image)
    cv2.waitKey(0)


def parse_args():
    parser = argparse.ArgumentParser()

    # only process image, video OR webcam 
    processing_group = parser.add_mutually_exclusive_group(required=True)
    processing_group.add_argument('-i', '--image', help="Input image path")
    processing_group.add_argument('-v', '--video', help="Input video path")
    processing_group.add_argument('-w', '--webcam', nargs='?', const=0, type=int, help="Input webcam number")

    # using the OpenCV haar cascade face classifier by default
    parser.add_argument('-x', '--xml', default="./haarcascade_frontalface_alt.xml", help="Classifer xml path")
    return parser.parse_args()


def main():
    args = parse_args()

    redactor = Redactor.fromXmlClassifier(args.xml)

    if args.image:
        process_image(args.image, redactor)
    elif args.video:
        process_video(args.video, redactor)
    elif args.webcam:
        process_webcam(args.webcam, redactor)


if __name__ == '__main__':
  main()


