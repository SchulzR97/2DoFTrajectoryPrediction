import numpy as np

class Transform_2DoF:
    def __init__(self, limb1, limb2, refX, refY):
        self.limb1 = limb1
        self.limb2 = limb2
        self.refX = refX
        self.refY = refY

    def to_euler(self, posX, posY):
        x = self.refX + posX
        y = self.refY + posY

        a = self.limb2
        b = self.limb1
        c = np.sqrt(x**2 + y**2)
        alpha = np.arccos((b**2 + c**2 - a**2) / (2*b*c))
        delta = np.arctan2(y, x)
        j1 = alpha + delta

        gamma = np.arccos((a**2 + b**2 - c**2) / (2*a*b))
        j2 = gamma

        return j1, j2

    def to_cartesian(self):
        pass

if __name__ == '__main__':
    ik = Transform_2DoF(10, 10, 0, 0)

    j1 = 45 
    j2 = 180
    phi1 = 180-90-j1
    phi2 = j2-90-phi1

    j1 = j1 / 180 * np.pi
    j2 = j2 / 180 * np.pi
    phi1 = phi1 / 180 * np.pi
    phi2 = phi2 / 180 * np.pi

    y1 = np.sin(j1) * ik.limb1
    y2 = np.sin(phi2) * ik.limb2

    posX = np.cos(j1) * ik.limb1 + np.cos(phi2) * ik.limb2
    posY = np.sin(j1) * ik.limb1 + np.sin(phi2) * ik.limb2
    j1_calc, j2_calc = ik.to_euler(posX, posY)

    print('j1', j1)
    print('j2', j2)
    print('j1_calc', j1_calc)
    print('j2_calc', j2_calc)
    pass