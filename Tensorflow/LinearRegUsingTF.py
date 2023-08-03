import tensorflow as tf
import pygame as pg 
import random
pg.init()
width = 600
height = 600
screen = pg.display.set_mode((width, height))
clock = pg.time.Clock() 
pg.display.set_caption('Linear Regression Using tensorflow')

running = True
bgcol = 50

x_vals = []
y_vals = []
ms = tf.Variable(random.uniform(0,1))
bs = tf.Variable(random.uniform(0,1))
m = ms.numpy()
b = bs.numpy()

def drawLine() : 
    global m , b
    x1 = 0 
    y1 = m*x1 + b 
    x2 = 1
    y2 = m*x2 + b 

    x1 = x1* width 
    y1 = (1-y1) * height
    x2 = x2* width 
    y2 = (1-y2) * height
    pg.draw.line(screen , (0,250,250), (x1,y1),(x2,y2),1)

def predict(x) : 
   xs = tf.Variable(x, dtype=tf.float32)
   ys = xs*ms + bs
   return ys
def loss(pred , labels) : 
   cost = tf.reduce_mean(tf.square(pred-labels))
   return cost
def scale(val , startX , endX , startY , endY) : 
    x1 , y1 = startX , startY
    x2 , y2 = endX , endY
    return y2 - (y2-y1)*(x2-val)/(x2-x1)
def mousePressed() : 
    mx , my = pg.mouse.get_pos()
    x = scale(mx , 0 , width , -1 , 1)
    y = scale(my , 0 , height , 1, -1)
    x_vals.append(x)
    y_vals.append(y)

learningRate = .5
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learningRate)

while running : 
    screen.fill((bgcol,bgcol,bgcol))
    if len(x_vals) > 1 : 
      for i in range(1) : 
          with tf.GradientTape() as tape:
            ys = tf.Variable(y_vals)
            test = predict(x_vals)
            errors = loss(test , ys)
          gradients = tape.gradient(errors , [ms , bs])
          ms.assign_sub(learningRate*gradients[0])
          bs.assign_sub(learningRate*gradients[1])
    #   m = ms.numpy()
    #   b = bs.numpy()
    #   drawLine()
      lineX = [-1,1]
      lineY = predict(lineX).numpy()
      x1 = scale(lineX[0] , -1 ,1 , 0 , width)
      x2 = scale(lineX[1] ,-1 ,1 , 0 , width)
      y1 = scale(lineY[0] , 1 , -1 , 0 , height)
      y2 = scale(lineY[1] , 1 , -1 , 0 , height)
      pg.draw.line(screen , ("blue") , (x1,y1),(x2,y2),1)
    
    if pg.mouse.get_pressed()[0]: mousePressed()
     
    for i in range(len(x_vals)):
        xpos = scale(x_vals[i] , -1 , 1 , 0 , width)
        ypos = scale(y_vals[i] , 1 , -1 , 0 , height)
        pg.draw.circle(screen , (250,250,250),(xpos,ypos),5)

    for event in pg.event.get() : 
        if event.type == pg.QUIT :  
            running = False 
        elif event.type == pg.KEYDOWN : 
            if event.key == pg.K_ESCAPE : 
                running = False 
            if event.key == pg.K_SPACE : 
                x_vals = []
                y_vals = []
               
    pg.display.flip()
    clock.tick(30)
pg.quit()
