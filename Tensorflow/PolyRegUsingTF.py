import tensorflow as tf
import pygame as pg 
import random

pg.init()
width = 600
height = 600
screen = pg.display.set_mode((width, height))
clock = pg.time.Clock() 
pg.display.set_caption('Polynomial Regression Using tensorflow')

running = True
bgcol = 50

x_vals = []
y_vals = []
a = tf.Variable(random.uniform(0,1))
b = tf.Variable(random.uniform(0,1))
c = tf.Variable(random.uniform(0,1))
d = tf.Variable(random.uniform(0,1))
def predict(x) : 
   xs = tf.Variable(x, dtype=tf.float32)
#    y = ax2 + bx + c
#    ys = tf.square(xs)*a + b*xs + c 
   ys = tf.math.pow(xs ,3) * a + tf.square(xs)*b + c*xs + d 
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

# @tf.function
# def train(opt, input):
#    with tf.GradientTape() as tape:
#         tape.watch(input)
#         loss = tf.reduce_mean(input)
#     gradients = tape.gradient(loss, input)
#     opt.apply_gradients(zip(gradients, input))

# opt = tf.keras.optimizers.Adam(learning_rate=1)
# input = tf.random.normal((1, 1))

# train(opt, input)

learningRate = .3

opt = tf.keras.optimizers.Adam(learning_rate=learningRate*2)

while running : 
    screen.fill((bgcol,bgcol,bgcol))
    if len(x_vals) > 1 : 
      for i in range(1) : 
          with tf.GradientTape() as tape:
            ys = tf.Variable(y_vals)
            test = predict(x_vals)
            errors = loss(test , ys)
          gradients = tape.gradient(errors , [a , b , c ,d])
          opt.apply_gradients(zip(gradients, [a,b,c ,d]))
          a.assign_sub(learningRate*gradients[0])
          b.assign_sub(learningRate*gradients[1])
          c.assign_sub(learningRate*gradients[2])
          d.assign_sub(learningRate*gradients[3])

    lineX = []
    for i in range(-20,21,1) : lineX.append(i/20)
    lineY = predict(lineX).numpy()
    for i in range(len(lineX)-1) : 
        x1 = scale(lineX[i] , -1 ,1 , 0 , width)
        x2 = scale(lineX[i+1] ,-1 ,1 , 0 , width)
        y1 = scale(lineY[i] , 1 , -1 , 0 , height)
        y2 = scale(lineY[i+1] , 1 , -1 , 0 , height)
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
