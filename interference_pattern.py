import numpy as np 
import matplotlib.pyplot as plt
import math 
  
amount_of_waves = 10
theta = np.linspace( 0 , 2 * np.pi , 150 ) 
figure, axes = plt.subplots( 1 )

def get_intersections(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d=math.sqrt((x1-x0)**2 + (y1-y0)**2)
    
    # non intersecting
    if d > r0 + r1 :
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d
        
        return (x3, y3, x4, y4)

def create_speakers_waves(positions, amount_of_waves):
    speakers_waves = []
    for pos in positions:
        speakers_waves.append(create_waves_for_speaker(pos, amount_of_waves))
    return speakers_waves

def create_waves_for_speaker(pos,amount_of_waves):
    waves_for_one_speaker = []
    for radius in range(amount_of_waves):
        x = radius * np.cos( theta ) + pos[0]
        y = radius * np.sin( theta ) + pos[1]
        waves_for_one_speaker.append((x,y))
    return waves_for_one_speaker

def draw_intersections(intersection_points):
    for intersection_point in intersection_points:
        print(intersection_point)
        axes.plot(intersection_point[0], intersection_point[1], marker='o', color='r')
        axes.plot(intersection_point[2], intersection_point[3], marker='o', color='r')

def draw_circles(circles):
    for index, circle in enumerate(circles):
        if index == 0:
            color = 'k'
        else:
            color = 'b'
        for x,y in circle:
            axes.plot( x, y,  color=color ) 

def draw_speakers(positions):
    for position in positions:
        axes.plot(position[0],position[1],marker='o', color='g')


positions = [(0,1),(0,0)]
circles = create_speakers_waves(positions, amount_of_waves)

def get_intersections_for_circles(positions, amount_of_waves):
    intersection_points = []
    for index, position in enumerate(positions):
        for radius in range(amount_of_waves):
            for radius2 in range(amount_of_waves):
                print(len(positions))
                if len(positions)-2>=index:
                    intersection = get_intersections(position[0],position[1],radius, positions[index+1][0], positions[index+1][1], radius2)
                    if intersection:
                        intersection_points.append(get_intersections(position[0],position[1],radius, positions[index+1][0], positions[index+1][1], radius2))
    return intersection_points

intersections = get_intersections_for_circles(positions, amount_of_waves)

draw_circles(circles)
draw_intersections(intersections)
draw_speakers(positions)
#draw_sound_waves(5)
#draw_sound_waves(0)
plt.title( 'Speaker setup' ) 
plt.show()