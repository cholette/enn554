from .math import vecnorm
from numpy import array

def heliostat_normal(helio_pos,aimpoint,sun_pos):
    
    xa,ya,za = aimpoint
    xh,yh,zh = helio_pos
    helio_to_receiver = array([xa-xh,ya-yh,za-zh])
    helio_to_receiver /= vecnorm(helio_to_receiver)
    sun_pos /= vecnorm(sun_pos)
    N = sun_pos+helio_to_receiver
    return N/vecnorm(N)