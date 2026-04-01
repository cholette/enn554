import numpy as np
from dataclasses import dataclass

def sind(x):
    return np.sin(np.deg2rad(x))

def cosd(x): 
    return np.cos(np.deg2rad(x))

def tand(x):
    return np.tan(np.deg2rad(x))

def acosd(x):
    return np.rad2deg( np.arccos(x) )

def asind(x):
    return np.rad2deg( np.arcsin(x) )

def atand(x):
    return np.rad2deg( np.arctan(x) )

def vecnorm(x):
    return np.sqrt(np.dot(x,x))

def proj(a,b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a,b)/vecnorm(b) * b

@dataclass
class Circle:
    x: float
    y: float
    r: float

    def area(self):
        return np.pi*self.r**2

def circle_overlap_area(c0:Circle,c1:Circle) -> float:
    
    dx,dy = c0.x-c1.x, c0.y-c1.y
    d = np.sqrt(dx**2+dy**2)
    r0,r1 = c0.r,c1.r
    
    # cases
    if d > r0+r1:
        area = 0
    elif d <= np.abs(r0-r1):
        area = np.pi * np.min([r0,r1])**2
    else:
        a = (r0**2 - r1**2 + d**2)/(2*d)
        # b = d - a
        h = np.sqrt(r0**2 - a**2)
        ϕ_0 = np.arcsin(h/r0)
        ϕ_1 = np.arcsin(h/r1)
        area = r0**2 * ϕ_0 + r1**2* ϕ_1 - d*h

    return area

    