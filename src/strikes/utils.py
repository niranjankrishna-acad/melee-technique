import math
class Landmark:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"{self.x} {self.y} {self.z}"

    @staticmethod
    def from_landmark(landmark):
        return Landmark(landmark.x, landmark.y, landmark.z)

    @staticmethod
    def sub(landmark_1, landmark_2):
        neg_landmark_2 = Landmark.neg(landmark_2)
        return Landmark.add(landmark_1, neg_landmark_2)

    @staticmethod
    def add(landmark_1, landmark_2):
        return Landmark(
        x=landmark_1.x - landmark_2.x,
        y=landmark_1.y - landmark_2.y,
        z=landmark_1.z - landmark_2.z
    )

    @staticmethod
    def neg(landmark):
        return Landmark(
        x=-1 * landmark.x,
        y=-1 * landmark.y,
        z=-1 * landmark.z
    )

    @staticmethod
    def dist(landmark_1, landmark_2):
        return math.sqrt((landmark_1.x - landmark_2.x)**2 + (landmark_1.y - landmark_2.y)**2 + (landmark_1.z - landmark_2.z)**2)
    
    @staticmethod
    def distx(landmark_1, landmark_2):
        return landmark_1.x - landmark_2.x

    @staticmethod
    def disty(landmark_1, landmark_2):
        return landmark_1.y - landmark_2.y

    @staticmethod
    def distz(landmark_1, landmark_2):
        return landmark_1.z - landmark_2.z