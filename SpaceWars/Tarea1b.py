import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
from PIL import Image

import transformations as tr
import basic_shapes as bs
import easy_shaders as es

class GPUShape:
    def __init__(self):
        self.vao = 0
        self.vbo = 0
        self.ebo = 0
        self.texture = 0
        self.size = 0


# A class to store the application control
class Controller:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta=0
        self.Attack = False
        self.fillPolygon = True
        

controller = Controller()

def on_key(window, key, scancode, action, mods):
    global controller
    
    if action == glfw.REPEAT or action == glfw.PRESS:
        controller.x += 0
    
    if action != glfw.PRESS:
        return
    
    elif key==glfw.KEY_ENTER:
        controller.fillPolygon = not controller.fillPolygon
        
    elif key==glfw.KEY_SPACE:
        controller.Attack = not controller.Attack
        
    elif key == glfw.KEY_A:
        if controller.x>=-0.75:
            controller.x -= 0.1

    elif key == glfw.KEY_D:
        if controller.x<=0.75:
            controller.x += 0.1

    elif key == glfw.KEY_W:
        if controller.y<=0.5:
            controller.y += 0.1

    elif key == glfw.KEY_S:
        if controller.y>=0.1 :
            controller.y -= 0.1
          

    elif key == glfw.KEY_ESCAPE:
        sys.exit()

    else:
        print('Unknown key')



def createNaveP(color1,color2,color3):
    # Here the new shape will be stored
    gpuShape = GPUShape()

    # Defining locations and colors for each vertex of the shape
    V1_0=[-0.5,-0.5 ]
    V2_0=[0.5 , -0.5]
    V3_0=[0.5 ,  0.5]
    V4_0=[-0.5, 0.5 ]
    original=np.array([V1_0,V2_0,V3_0,V4_0])
    
    # transformaciones necesarias para realizar el rey
    transforms=[tr.matmul([tr.scale(0.2,0.7,1)]),
        tr.matmul([tr.translate(-0.15,0,0),tr.scale(1/10,0.4,1)]),# cuerpo
        tr.matmul([tr.translate(-0.25,-0.1,0),tr.scale(1/10,0.4,1)]),
        tr.matmul([tr.translate(-0.35,-0.2,0),tr.scale(1/10,0.4,1)]),
        tr.matmul([tr.translate(0.15,0,0),tr.scale(1/10,0.4,1)]),
        tr.matmul([tr.translate(0.25,-0.1,0),tr.scale(1/10,0.4,1)]),
        tr.matmul([tr.translate(0.35,-0.2,0),tr.scale(1/10,0.4,1)]), #cuerpo
        tr.matmul([tr.translate(0.425,-0.1,0),tr.scale(1/15,0.3,1)]), #arma
        tr.matmul([tr.translate(-0.425,-0.1,0),tr.scale(1/15,0.3,1)]), #arma
        tr.matmul([tr.translate(0,0.4,0),tr.scale(1/15,1/10,1)]), #cabeza
        tr.matmul([tr.translate(0.425,0.08,0),tr.scale(1/15,1/15,1)]), #arma cuadrado
        tr.matmul([tr.translate(-0.425,0.08,0),tr.scale(1/15,1/15,1)])] #arma cuadrado
        
    Vertex=[]
    Matrices=[]
    for transform in transforms:  
        V=applyTransform(transform, original)
        Matrices.append(V)
    for k in range(0,7):
        M=Matrices[k]
        for i in range(len(M)):
            for j in range(0,2):
                Vertex.append(M[i,j])# guardo las cordenada en una lista
            Vertex.append(0)
            Vertex.append(color1[0])
            Vertex.append(color1[1])
            Vertex.append(color1[2])
    for k in range(7,10):
        M=Matrices[k]
        for i in range(len(M)):
            for j in range(0,2):
                Vertex.append(M[i,j])# guardo las cordenada en una lista
            Vertex.append(0)
            Vertex.append(color2[0])
            Vertex.append(color2[1])
            Vertex.append(color2[2])
    
    for k in range(10,12):
        M=Matrices[k]
        for i in range(len(M)):
            for j in range(0,2):
                Vertex.append(M[i,j])# guardo las cordenada en una lista
            Vertex.append(0)
            Vertex.append(color3[0])
            Vertex.append(color3[1])
            Vertex.append(color3[2])
        
    vertices = Vertex

        
    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices =[0, 1, 2,
         2, 3, 0, 

         4, 5, 6,
         6, 7, 4,

         8, 9, 10,
         10,11,8,

         12 ,13,14,
         14, 15,12,

         16, 17,18,
         18, 19,16,

         20, 21,22,
         22, 23,20,

         24, 25,26,
         26, 27,24,
         
         28, 29, 30,
         30, 31, 28,

         32, 33, 34,
         34, 35, 32,

         36, 37, 38,
         38, 39, 36,

        40, 41, 42,
        42, 43, 40,

        44, 45, 46,
        46, 47 ,44
         ]
    return bs.Shape(vertices, indices)


def createNaveE(color1,color2,color3):
    # Here the new shape will be stored
    gpuShape = GPUShape()

    # Defining locations and colors for each vertex of the shape
    V1_0=[-0.5,-0.5 ]
    V2_0=[0.5 , -0.5]
    V3_0=[0.5 ,  0.5]
    V4_0=[-0.5, 0.5 ]
    original=np.array([V1_0,V2_0,V3_0,V4_0])
    
    # transformaciones necesarias para realizar el rey
    transforms=[tr.matmul([tr.translate(0,0.1,0),tr.scale(0.2,0.4,1)]),
        tr.matmul([tr.translate(-0.12,0,0),tr.scale(1/10,0.4,1)]),# cuerpo
        tr.matmul([tr.translate(-0.22,-0.1,0),tr.scale(1/10,0.4,1)]),
        tr.matmul([tr.translate(-0.32,-0.2,0),tr.scale(1/10,0.4,1)]),
        tr.matmul([tr.translate(0.12,0,0),tr.scale(1/10,0.4,1)]),
        tr.matmul([tr.translate(0.22,-0.1,0),tr.scale(1/10,0.4,1)]),
        tr.matmul([tr.translate(0.32,-0.2,0),tr.scale(1/10,0.4,1)]), #cuerpo
        tr.matmul([tr.translate(0.4,-0.3,0),tr.scale(1/15,0.3,1)]), #arma
        tr.matmul([tr.translate(-0.4,-0.3,0),tr.scale(1/15,0.3,1)]), #arma
        tr.matmul([tr.translate(0,0.35,0),tr.scale(1/15,1/10,1)]), #cabeza
        tr.matmul([tr.translate(0.4,-0.18,0),tr.scale(1/15,1/15,1)]), #arma cuadrado
        tr.matmul([tr.translate(-0.4,-0.18,0),tr.scale(1/15,1/15,1)])] #arma cuadrado
        
    Vertex=[]
    Matrices=[]
    for transform in transforms:  
        V=applyTransform(transform, original)
        Matrices.append(V)
    for k in range(0,7):
        M=Matrices[k]
        for i in range(len(M)):
            for j in range(0,2):
                Vertex.append(M[i,j])# guardo las cordenada en una lista
            Vertex.append(0)
            Vertex.append(color1[0])
            Vertex.append(color1[1])
            Vertex.append(color1[2])
    for k in range(7,10):
        M=Matrices[k]
        for i in range(len(M)):
            for j in range(0,2):
                Vertex.append(M[i,j])# guardo las cordenada en una lista
            Vertex.append(0)
            Vertex.append(color2[0])
            Vertex.append(color2[1])
            Vertex.append(color2[2])
    
    for k in range(10,12):
        M=Matrices[k]
        for i in range(len(M)):
            for j in range(0,2):
                Vertex.append(M[i,j])# guardo las cordenada en una lista
            Vertex.append(0)
            Vertex.append(color3[0])
            Vertex.append(color3[1])
            Vertex.append(color3[2])
        
    vertices = Vertex


    indices = [0, 1, 2,
         2, 3, 0, 

         4, 5, 6,
         6, 7, 4,

         8, 9, 10,
         10,11,8,

         12 ,13,14,
         14, 15,12,

         16, 17,18,
         18, 19,16,

         20, 21,22,
         22, 23,20,

         24, 25,26,
         26, 27,24,
         
         28, 29, 30,
         30, 31, 28,

         32, 33, 34,
         34, 35, 32,

         36, 37, 38,
         38, 39, 36,

        40, 41, 42,
        42, 43, 40,

        44, 45, 46,
        46, 47 ,44
         ]

    return bs.Shape(vertices, indices)


def applyTransform(transform, vertices):

    # Creating an array to store the transformed vertices
    # Since we will replace its content, the initial data is just garbage memory.
    transformedVertices = np.ndarray((len(vertices), 2), dtype=float)

    for i in range(len(vertices)):
        vertex2d = vertices[i]
        # input vertex only has x,y
        # expresing it in homogeneous coordinates
        homogeneusVertex = np.array([vertex2d[0], vertex2d[1], 0.0, 1.0])
        transformedVertex = np.matmul(transform, homogeneusVertex)

        # we are not prepared to handle 3d in this example
        # converting the vertex back to 2d
        transformedVertices[i,0] = transformedVertex[0] / transformedVertex[3]
        transformedVertices[i,1] = transformedVertex[1] / transformedVertex[3]
        
    return transformedVertices

class Nave: # creo la clase nave 
    def __init__(self):
        self.vida=3  # defino su vida
        self.posx=0  # defino su posicion en x
        self.posy=0  # defino su posicion en y
        self.posxL=0 # defino la posicion en x de su rayo
        self.posyL=0 # defino la posicion en y de su rayo
        self.posxL2=0 # analogo a posxL 
        self.posyL2=0 # analogo a posyL 
        self.posxL3=0 # analogo a posxL 
        self.posyL3=0 # analogo a posxL 
        self.Attack=False  # defino si la nave esta atacando o no
        self.AttackP=[False, False,False] # analogo a Attack 
        self.Choque="fuera" # defino si tengo a alguien dentro 
        self.Impacto=False  # defino si impacte a alguien
        self.ImpactoP=[False, False,False] # analogo a impacto
        self.AreaInicial=np.array([[-0.5,-0.5 ],[0.5 , -0.5],[0.5, 0.5],[-0.5,0.5]]) #defino el area interna de una nave
        self.Area=np.array([[-0.5,-0.5 ],[0.5 , -0.5],[0.5, 0.5],[-0.5,0.5]]) #defino el area interna de una nave
        self.timer=-0.7 
        self.assassin=-1 # defino quien me mato , parte con -1 porque al inicio nadie lo ha hecho
    
    def MovX(self,x): # para la principal
        if x!=self.posx: # si el x que ingresa es distinto al que tengo guardado
            dif=x-self.posx
            self.posx=x
            return dif
        else:
            return 0
    def MovY(self,y): # para la principal
        if y!=self.posy: # si el y que ingresa es distinto al que tengo guardado
            dif=y-self.posy
            self.posy=y
            return dif
        else:
            return 0

      
            
    def Ataque(self,dt):# para la nave enemiga

        if not self.Attack: 
            self.posxL=self.posx   #guardo las posiciones de la nave en la posicion del rayo
            self.posyL=self.posy+0.1
            self.Attack=True

        if self.Attack:
            self.posyL+=dt
            if self.posyL>2 or  self.Impacto==True: # si mi rayo salio de la pantalla o impacto a alguien
                self.Attack=False




    def colision(self,x,y,Nave,n): #solo para la nave enemiga
        if self.Choque=="fuera" and Nave.ImpactoP[n]==False: # si no tengo un rayo dentro y la otra nave no a impactado a nadie
            if x>self.Area[0][0]and x<self.Area[1][0]:
                if y>self.Area[0][1] and y<self.Area[2][1]:  # verifico que  este en mi area interna
                    self.Choque="dentro"  # tengo un rayo en mi interior
                    self.vida-=1          # me resto una vida
                    Nave.ImpactoP[n]=True # coloco true al rayo que me impacto
                    self.assassin=n       # guardo cual de los tres rayos de la nave principal me impacto
                    
        elif self.Choque=="dentro" and Nave.ImpactoP[n]==True: #si tengo un rayo dentro y la otra nave habia impactado
            if not (x>self.Area[0][0]and x<self.Area[1][0]):   # verifico que el rayo ya no este en mi area interna
                if not(y>self.Area[0][1] and y<self.Area[2][1]):
                    self.Choque="fuera"  # ya no tengo rayo en mi interior
                    
    def ciclo(self,dt): # para la nave enemiga
        if self.timer>=-0.7 and self.timer<0 : #para ingresar a la pantalla

            transform=tr.matmul([tr.translate( +dt/2 ,0, 0)])
            self.Area=applyTransform(transform,self.Area) # traslado el area interna  de la nave
            self.posx+=dt/2
            self.timer+=dt/2

        if self.timer>=0 and self.timer<1.6: # para moverla de izquierdo a la derecha

            transform=tr.matmul([tr.translate( +dt/2 ,0, 0)])
            self.Area=applyTransform(transform,self.Area)
            self.posx+=dt/2
            self.timer+=dt/2

        if self.timer>=1.6 and self.timer<2.2: #para moverla de arriba hacia abajo

            transform=tr.matmul([tr.translate( 0 ,-dt/2, 0)])
            self.Area=applyTransform(transform,self.Area)
            self.posy+=dt/2
            self.timer+=dt/2

        if self.timer>=2.2 and self.timer<3.8:# para moverla de la derecha a la izquierda
                
            transform=tr.matmul([tr.translate(-dt/2,0 , 0)])
            self.Area=applyTransform(transform,self.Area)
            self.posx-=dt/2
            self.timer+=dt/2

        if self.timer>=3.8 and self.timer<4.4:#para moverla de abajo hacia arriba

            transform=tr.matmul([tr.translate( 0,dt/2 , 0)])
            self.Area=applyTransform(transform,self.Area)
            self.posy-=dt/2
            self.timer+=dt/2
           
        if self.timer>=4.4:
            self.timer=0 # coloco el timer en 0 , para que repita el ciclo

    def jesus(self): # para la nave enemiga
        self.vida=3
        self.posx=0
        self.posy=0
        self.Choque="fuera"
        self.Area=self.AreaInicial 
        self.timer=-0.7
        self.assassin=-1

# creo la nave principal y defino su area interna
transformNavePrincipal=tr.matmul([tr.translate(0.0,-0.82,0),tr.scale(0.3,0.2,0)])

NaveP=Nave()
NaveP.Area=applyTransform(transformNavePrincipal,NaveP.Area)
NaveP.AreaInicial=applyTransform(transformNavePrincipal,NaveP.AreaInicial)

#creo las naves enemigas y definos sus areas internas
transformNavesEnemigas=tr.matmul([tr.translate(-1.5,0.8,0),tr.translate(0,0.03,0),tr.scale(0.28,0.2,1)])

NaveE=Nave()
NaveE.Area=applyTransform(transformNavesEnemigas,NaveE.Area)
NaveE.AreaInicial=applyTransform(transformNavesEnemigas,NaveE.AreaInicial)

NaveE2=Nave()
NaveE2.Area=applyTransform(transformNavesEnemigas,NaveE2.Area)
NaveE2.AreaInicial=applyTransform(transformNavesEnemigas,NaveE2.AreaInicial)

NaveE3=Nave()
NaveE3.Area=applyTransform(transformNavesEnemigas,NaveE3.Area)
NaveE3.AreaInicial=applyTransform(transformNavesEnemigas,NaveE3.AreaInicial)

Enemy=[NaveE,NaveE2,NaveE3]

def main():
    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 600
    height = 600

    window = glfw.create_window(width, height, "Space War", None, None)

    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # A simple shader program with position and texture coordinates as inputs.
    pipelineTexture = es.SimpleTextureTransformShaderProgram()
    pipelineNoTexture = es.SimpleTransformShaderProgram()
    # Telling OpenGL to use our shader program

    # Setting up the clear screen color
    glClearColor(0.25, 0.25, 0.25, 1.0)

    # Enabling transparencies
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Creating shapes on GPU memory

    gpuFondo= es.toGPUShape(bs.createTextureQuad("espacio.jpg"), GL_REPEAT, GL_NEAREST)
    gpuWin= es.toGPUShape(bs.createTextureQuad("Win.png"), GL_REPEAT, GL_NEAREST)
    gpuFin= es.toGPUShape(bs.createTextureQuad("GameOver.jpg"), GL_REPEAT, GL_NEAREST)
    gpuQuadP=es.toGPUShape(bs.createColorQuad(1,128/255,0))
    gpuQuadE=es.toGPUShape(bs.createColorQuad(65/255,105/255,2))
    gpuNave=es.toGPUShape(createNaveP([1,1,1],[1,128/255,0],[1,1,1]))
    gpuNaveE=es.toGPUShape(createNaveE([155/255,155/255,155/255],[65/255,105/255,1],[1,1,1]))

    t0 = glfw.get_time()
    t=0
    n =int(sys.argv[1])
    contador=n-3

    while not glfw.window_should_close(window):


        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1
        t+=dt
        # Using GLFW to check for input events
        glfw.poll_events()

        if controller.fillPolygon:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            
        else:
            
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT)

    

        
    ##################### Fondo del juego ##################################
        glUseProgram(pipelineTexture.shaderProgram)

        
        transform=tr.matmul([tr.translate(0,-t/5,0),tr.scale(2,2,1)])
        pipelineTexture.drawShape( gpuFondo, transform)

        transform=tr.matmul([tr.translate(0,2-t/5,0) ,tr.scale(1,-1,1),tr.scale(2,2,1)])
        pipelineTexture.drawShape( gpuFondo, transform)

        if t/5>1 :

            transform=tr.matmul([tr.translate(0,4-t/5,0),tr.scale(2,2,1)])
            pipelineTexture.drawShape( gpuFondo, transform)

            transform=tr.matmul([tr.translate(0,6-t/5,0) ,tr.scale(1,-1,1),tr.scale(2,2,1)])
            pipelineTexture.drawShape( gpuFondo, transform)

            transform=tr.matmul([tr.translate(0,8-t/5,0) ,tr.scale(2,2,1)])
            pipelineTexture.drawShape( gpuFondo, transform)
    
        if t/5>=8:
            t=0    
        
    ########################## Jugador ###################################

        glUseProgram(pipelineNoTexture.shaderProgram)

    ######################### Rayo laser ################################

        
        if not controller.Attack and NaveP.AttackP[0]==False: # si el usuario no a apretado space y el primer rayo no esta atacando
            NaveP.posxL=controller.x    # guardo la posicion de la nave como la posicion del laser
            NaveP.posyL=-0.7+controller.y

        if not controller.Attack and NaveP.AttackP[1]==False:  # si el usuario no a apretado space y el segundo rayo no esta atacando
            NaveP.posxL2=controller.x   # guardo la posicion de la nave como la posicion del laser
            NaveP.posyL2=-0.7+controller.y
        
        if not controller.Attack and NaveP.AttackP[2]==False:  # si el usuario no a apretado space y el tercer rayo no esta atacando
            NaveP.posxL3=controller.x   # guardo la posicion de la nave como la posicion del laser
            NaveP.posyL3=-0.7+controller.y


        if controller.Attack and NaveP.AttackP[0]==False: # el usuario apreta SPACE y el primer rayo no esta atacando
            NaveP.AttackP[0]=True
            controller.Attack=False
        
        if controller.Attack and NaveP.AttackP[0]==True and NaveP.AttackP[1]==False:# el usuario apreta SPACE y el primer rayo esta atacando
            NaveP.AttackP[1]=True                                                   # pero el segundo rayo no esta atacando
            controller.Attack=False
          
        if controller.Attack and NaveP.AttackP[0]==True and NaveP.AttackP[1]==True and NaveP.AttackP[2]==False:#el usuario apreta Space y el
            NaveP.AttackP[2]=True                                                                              # primero como el segundo rayo 
            controller.Attack=False                                                                            #  estan atacando pero el terce
                                                                                                               # no lo esta


        if controller.Attack and NaveP.AttackP[0]==True and NaveP.AttackP[1]==True and NaveP.AttackP[2]==True:#el usuario apreta Space y los                                                                         # primero como el segundo rayo 
            controller.Attack=False                                                                            # tres rayos estan atacando
                                                                                                            
        

        if NaveP.AttackP[0]==True:
            NaveP.posyL+=dt
            transform=tr.matmul([tr.translate(NaveP.posxL,+NaveP.posyL,0),tr.scale(1/30,1/10,1)])
            pipelineNoTexture.drawShape( gpuQuadP, transform)
            if NaveP.posyL>1 or NaveP.ImpactoP[0]==True :  # si el rayo salio de la pantalla o impacto a alguien
                NaveP.posyL=-0.7
                NaveP.AttackP[0]=False
                NaveP.ImpactoP[0]=False

        if NaveP.AttackP[1]==True:
            NaveP.posyL2+=dt
            transform=tr.matmul([tr.translate(NaveP.posxL2,+NaveP.posyL2,0),tr.scale(1/30,1/10,1)])
            pipelineNoTexture.drawShape( gpuQuadP, transform) 
            if NaveP.posyL2>1 or NaveP.ImpactoP[1]==True :   # si el rayo salio de la pantalla o impacto a alguien
                NaveP.posyL2=-0.7
                NaveP.AttackP[1]=False
                NaveP.ImpactoP[1]=False
        
        if NaveP.AttackP[2]==True:
            NaveP.posyL3+=dt
            transform=tr.matmul([tr.translate(NaveP.posxL3,+NaveP.posyL3,0),tr.scale(1/30,1/10,1)])
            pipelineNoTexture.drawShape( gpuQuadP, transform)
            if NaveP.posyL3>1 or NaveP.ImpactoP[2]==True :   # si el rayo salio de la pantalla o impacto a alguien
                NaveP.posyL3=-0.7
                NaveP.AttackP[2]=False 
                NaveP.ImpactoP[2]=False

        ######################## Area interna  #########################
 
        transform=tr.translate(NaveP.MovX(controller.x),NaveP.MovY(controller.y),0)
        NaveP.Area=applyTransform(transform,NaveP.Area) #traslado el area interna de la nave 


         ####################  colisiones al jugador #######################3
        for nave in Enemy:
            if NaveP.Choque=="fuera" and nave.Impacto==False: #si no tengo un rayo dentro y la otra nave no le a dado a nadie
                if -1.5 +nave.posxL>NaveP.Area[0][0]and -1.5 +nave.posxL<NaveP.Area[1][0]: # verifico si esta en mi area interna
                    if 0.8-nave.posyL>NaveP.Area[0][1] and 0.8-nave.posyL<NaveP.Area[2][1]:
                        NaveP.Choque="dentro" # tengo un rayo en mi interior
                        nave.Impacto=True
                        NaveP.vida-=1 # me quito una vida

            elif NaveP.Choque=="dentro" and nave.Impacto==True:#si tengo un rayo en mi interior y la otra nave impacto a alguien
                if not (-1.5 +nave.posxL>NaveP.Area[0][0] and -1.5 +nave.posxL<NaveP.Area[1][0]):# verifico que salio de mi area
                    if not(0.8-nave.posyL>NaveP.Area[0][1] and 0.8-nave.posyL<NaveP.Area[2][1]):
                        NaveP.Choque="fuera" # ya no tengo nadie dentro
                        nave.Impacto=False
                        
     ###################  observacion de mi area interna ################################

        #esto lo usaba para ver el area interna de mi nave principal
        #transform=tr.matmul([tr.translate(controller.x,controller.y,0),tr.translate(-0.0,-0.82,0),tr.scale(0.3,0.2,0)])
        #pipelineNoTexture.drawShape( gpuQuadE, transform)
                        
    ############################## Enemigos  ############################################

    ############ Rayo laser ###############################################
        
        for nave in Enemy:
            if t1>4:
                nave.Ataque(dt) # activo su ataque
                transform=tr.matmul([tr.translate(-1.5 +nave.posxL,0.8-nave.posyL, 0),tr.scale(1/30,1/10,1)])
                pipelineNoTexture.drawShape( gpuQuadE, transform)
            if nave.vida<3:
                nave.posx=-10 #si la nave esta muerta traslado su posicion lejos de la pantalla ,
                              # para que a la siguiente iteracion el rayo de esta no se vea
      

        ######## Detector de golpe de nave Enemiga #################

            nave.colision(NaveP.posxL,NaveP.posyL,NaveP,0)
            nave.colision(NaveP.posxL2,NaveP.posyL2,NaveP,1)
            nave.colision(NaveP.posxL3,NaveP.posyL3,NaveP,2)

        ##################### Movimiento y marco interno ##################################################
        #con esto veia el marco interno de mi nave

        #transform=tr.matmul([tr.translate(-0.8,0.8,0),tr.translate(0,0.03,0),tr.scale(0.28,0.2,1)])
        #pipelineNoTexture.drawShape(gpuQuadP, transform)
        #transform=tr.matmul([tr.translate(-0.8,0.8,0),tr.scale(1/3,1/3,1),tr.rotationZ(3.14)])
        #pipelineNoTexture.drawShape(gpuNaveE, transform)

        ################################# Dibujar Naves ################################################

        if NaveP.vida>0:# si aun tengo visa
            transform=tr.matmul([tr.translate(controller.x,controller.y,0),tr.translate(0,-0.8,0),tr.scale(1/3,1/3,1)])
            pipelineNoTexture.drawShape( gpuNave, transform)
            
        if NaveE.vida>2:# si aun tengo vida
            NaveE.ciclo(dt)
            transform=tr.matmul([tr.translate( -1.5 +NaveE.posx,0.8-NaveE.posy, 0),tr.scale(1/3,1/3,1),tr.rotationZ(3.14)])
            pipelineNoTexture.drawShape(gpuNaveE, transform) 

        if NaveE2.vida>2 and t1>4 and n>=2:# si aun tengo vida
            NaveE2.ciclo(dt)
            transform=tr.matmul([tr.translate( -1.5 +NaveE2.posx,0.8-NaveE2.posy, 0),tr.scale(1/3,1/3,1),tr.rotationZ(3.14)])
            pipelineNoTexture.drawShape(gpuNaveE, transform)

        if NaveE3.vida>2 and t1>6 and n>=3:# si aun tengo vida
            NaveE3.ciclo(dt)
            transform=tr.matmul([tr.translate( -1.5 +NaveE3.posx,0.8-NaveE3.posy, 0),tr.scale(1/3,1/3,1),tr.rotationZ(3.14)])
            pipelineNoTexture.drawShape(gpuNaveE, transform) 


        #####################resurreccion de naves enemigas ###########################################
        
        if NaveE.vida<3 and NaveP.ImpactoP[NaveE.assassin]==False and NaveE.Attack==False: 
            NaveE.Area=applyTransform(tr.matmul([tr.translate(-10,0,0)]),NaveE.Area)  # muevo el area interna de la nave mientras no sea resucitada    
            if contador>0: #si aun se necesitan naves 
                if NaveE2.timer>0 and NaveE2.timer<3.8 and NaveE3.timer>0 and NaveE3.timer<3.8:# verifico que cuando aparezca no choque con otra nave
                    contador-=1
                    NaveE.jesus() #resucito la nave

        if NaveE2.vida<3 and NaveP.ImpactoP[NaveE2.assassin]==False: 
            NaveE2.Area=applyTransform(tr.matmul([tr.translate(-10,0,0)]),NaveE2.Area)# muevo el area interna de la nave mientras no sea resucitada
            if contador>0:#si aun se necesitan naves
                if NaveE.timer>0 and NaveE.timer<3.8 and NaveE3.timer>0 and NaveE3.timer<3.8:# verifico que cuando aparezca no choque con otra nave
                    contador-=1
                    NaveE2.jesus() #resucito la nave


        if NaveE3.vida<3 and NaveP.ImpactoP[NaveE3.assassin]==False: 
            NaveE3.Area=applyTransform(tr.matmul([tr.translate(-10,0,0)]),NaveE3.Area)# muevo el area interna de la nave mientras no sea resucitada
            if contador>0:#si aun se necesitan naves
                if NaveE2.timer>0 and NaveE2.timer<3.8 and NaveE.timer>0 and NaveE.timer<3.8:# verifico que cuando aparezca no choque con otra nave
                    contador-=1
                    NaveE3.jesus() #resucito la nave
        if NaveE.vida<2 and  NaveE2.vida<2 and  NaveE3.vida<2:# si las tres naves murieron 
            if contador>0:#si aun se necesitan naves
                contador-=1
                NaveE.jesus() # resucito una

        ################ fin #####################
        glUseProgram(pipelineTexture.shaderProgram)
        if n==1 and NaveE.vida<3:
            pipelineTexture.drawShape( gpuWin,tr.identity()) 

        if n==2 and NaveE.vida<3 and NaveE2.vida<3 :
            pipelineTexture.drawShape( gpuWin,tr.identity()) 
        
        if n==3 and NaveE.vida<3 and NaveE2.vida<3 and NaveE3.vida<3 :
            pipelineTexture.drawShape( gpuWin,tr.identity()) 

        if NaveP.vida<=0: # si la nave principal murio ,nel jugador perdio
            pipelineTexture.drawShape( gpuFin, tr.identity())

        if contador==0 and NaveE.vida<3 and NaveE2.vida<3 and NaveE3.vida<3:# si no quedan naves enemigas y el contador esta en 0 , el jugador gano
            pipelineTexture.drawShape( gpuWin,tr.identity()) 

        glfw.swap_buffers(window)
    glfw.terminate()

main()
    
  

