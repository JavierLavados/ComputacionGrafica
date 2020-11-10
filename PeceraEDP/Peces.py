# coding=utf-8
"""
Daniel Calderon, CC3501, 2019-2
Projections example
"""

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys

import transformations as tr
import basic_shapes as bs
import easy_shaders as es
import scene_graph as sg
import lighting_shaders as ls
import local_shapes as l


PROJECTION_ORTHOGRAPHIC = 0
PROJECTION_FRUSTUM = 1
PROJECTION_PERSPECTIVE = 2


# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.projection = PROJECTION_ORTHOGRAPHIC


# We will use the global controller as communication with the callback function
controller = Controller()


def on_key(window, key, scancode, action, mods):

    if action != glfw.PRESS:
        return
    
    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_1:
        print('Orthographic projection')
        controller.projection = PROJECTION_ORTHOGRAPHIC

    elif key == glfw.KEY_2:
        print('Frustum projection')
        controller.projection = PROJECTION_FRUSTUM

    elif key == glfw.KEY_3:
        print('Perspective projection')
        controller.projection = PROJECTION_PERSPECTIVE

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)


import numpy as np
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D


def generateT(t):
    return np.array([[1, t, t**2, t**3]]).T


def hermiteMatrix(P1, P2, T1, T2):
    
    # Generate a matrix concatenating the columns
    G = np.concatenate((P1, P2, T1, T2), axis=1)

    # Hermite base matrix is a constant
    Mh = np.array([[1, 0, -3, 2], [0, 0, 3, -2], [0, 1, -2, 1], [0, 0, -1, 1]])    
    
    return np.matmul(G, Mh)


def bezierMatrix(P0, P1, P2, P3):
    
    # Generate a matrix concatenating the columns
    G = np.concatenate((P0, P1, P2, P3), axis=1)

    # Bezier base matrix is a constant
    Mb = np.array([[1, -3, 3, -1], [0, 3, -6, 3], [0, 0, 3, -3], [0, 0, 0, 1]])
    
    return np.matmul(G, Mb)


def plotCurve(ax, curve, label, color=(0,0,1)):
    
    xs = curve[:, 0]
    ys = curve[:, 1]
    zs = curve[:, 2]
    
    ax.plot(xs, ys, zs, label=label, color=color)
    

# M is the cubic curve matrix, N is the number of samples between 0 and 1
def evalCurve(M, N):
    # The parameter t should move between 0 and 1
    ts = np.linspace(0.0, 1.0, N)
    
    # The computed value in R3 for each sample will be stored here
    curve = np.ndarray(shape=(N, 3), dtype=float)
    
    for i in range(len(ts)):
        T = generateT(ts[i])
        curve[i, 0:3] = np.matmul(M, T).T
        
    return curve

def createAleta(N):
    
    R0L = np.array([[0, 0.5, 0]]).T
    R1L = np.array([[-0.5, 0.1, 0]]).T
    R2L = np.array([[-0.3, -0.5, 0]]).T
    R3L = np.array([[0, -0.5, 0]]).T
    
    GMbL = bezierMatrix(R0L, R1L, R2L, R3L)  # Puntos de control
    
    bezierCurveLeft = evalCurve(GMbL, N)
    
    R0R = np.array([[0, 0.5, 0]]).T
    R1R = np.array([[0.5, 0.1, 0]]).T
    R2R = np.array([[0.3, -0.5, 0]]).T
    R3R = np.array([[0, -0.5, 0]]).T
    
    GMbR = bezierMatrix(R0R, R1R, R2R, R3R)  # Puntos de control
    bezierCurveRight = evalCurve(GMbR, N)
    vertexDataList = []
    indicesList = []
    
    for index in range(len(bezierCurveLeft)):
        vertex = bezierCurveLeft[index]
        vertexDataList.extend(vertex)
        vertexDataList.extend([0, 0, 0])
        vertex = bezierCurveRight[index]
        vertexDataList.extend(vertex)
        vertexDataList.extend([0, 0, 0])
        
    for index in range(2*len(bezierCurveLeft)):
        indicesList += [index, index + 1, index + 2]
    
    # Here the new shape will be stored
    return bs.Shape(vertexDataList,indicesList)

def createAleta2(N):
    
    R0L = np.array([[0, 0.5, 0]]).T
    R1L = np.array([[-0.5, 0.1, 0]]).T
    R2L = np.array([[-0.6, -0.5, 0]]).T
    R3L = np.array([[-0.2, 0.7 , 0]]).T
    R4L = np.array([[0, 0, 0]]).T
    
    GMbL = bezierMatrix(R0L, R1L, R2L, R3L)  # Puntos de control
    
    bezierCurveLeft = evalCurve(GMbL, N)
    
    R0R = np.array([[0, 0.5, 0]]).T
    R1R = np.array([[0.5, 0.1, 0]]).T
    R2R = np.array([[0.6, -0.7, 0]]).T
    R3L = np.array([[0.2, 0.8, 0]]).T
    R3R = np.array([[0, 0, 0]]).T
    
    GMbR = bezierMatrix(R0R, R1R, R2R, R3R)  # Puntos de control
    
    bezierCurveRight = evalCurve(GMbR, N)
    
    
    vertexDataList = []
    indicesList = []
    
    for index in range(len(bezierCurveLeft)):
        vertex = bezierCurveLeft[index]
        vertexDataList.extend(vertex)
        vertexDataList.extend([0, 0, 0])
        vertex = bezierCurveRight[index]
        vertexDataList.extend(vertex)
        vertexDataList.extend([0, 0, 0])
        
    for index in range(2*len(bezierCurveLeft)):
        indicesList += [index, index + 1, index + 2]
    
    # Here the new shape will be stored
    return bs.Shape(vertexDataList,indicesList)



def createPezA():

    GpuCuerpo=esfera=es.toGPUShape(l.esfera(30,30,[57/255, 1, 20/255]))
    GpuOjo=es.toGPUShape(l.generateCirculo(15))
    GpuAletas=es.toGPUShape(createAleta(20))

    Cuerpo = sg.SceneGraphNode("Cuerpo")
    Cuerpo.transform=tr.matmul([tr.translate(0,0,0.9),tr.scale(2,0.6,1)])
    Cuerpo.childs+=[GpuCuerpo]

    Body = sg.SceneGraphNode("Body")
    Body.childs+=[Cuerpo]

    Cola1 = sg.SceneGraphNode("Cola1")
    Cola1.transform=tr.matmul([tr.rotationY(-2),tr.rotationX(3.14/2),tr.scale(1.4,2,1)])
    Cola1.childs+=[GpuAletas]

    Cola2 = sg.SceneGraphNode("Cola2")
    Cola2.transform=tr.matmul([tr.rotationY(5),tr.rotationX(3.14/2),tr.scale(1.4,2,1)])
    Cola2.childs+=[GpuAletas]

    ColaMov1 = sg.SceneGraphNode("ColaMov1")
    ColaMov1.childs+=[Cola1]

    ColaMov2 = sg.SceneGraphNode("ColaMov2")
    ColaMov2.childs+=[Cola2]

    ColaF1 = sg.SceneGraphNode("ColaF1")
    ColaF1.transform=tr.translate(-2.5,0,0.6)
    ColaF1.childs+=[ColaMov1]

    ColaF2 = sg.SceneGraphNode("ColaF2")
    ColaF2.transform=tr.translate(-2.5,0,1),
    ColaF2.childs+=[ColaMov2]

    ColaF = sg.SceneGraphNode("ColaF")
    ColaF.childs+=[ColaF1]
    ColaF.childs+=[ColaF2]

    Cresta1 = sg.SceneGraphNode("Cresta1")
    Cresta1.transform=tr.matmul([tr.translate(0.5,0,1.5),tr.rotationY(2.8),tr.rotationX(3.14/2),tr.scale(1.4,2,1)])
    Cresta1.childs+=[GpuAletas]

    Cresta2 = sg.SceneGraphNode("Cresta2")
    Cresta2.transform=tr.matmul([tr.translate(0.1,0,1.3),tr.rotationY(2.8),tr.rotationX(3.14/2),tr.scale(1.4,2,1)])
    Cresta2.childs+=[GpuAletas]

    Aleta1 = sg.SceneGraphNode("Aleta1")
    Aleta1.transform=tr.matmul([tr.translate(0,-0.6,0.2),tr.rotationZ(0.5),tr.rotationY(4),tr.rotationX(3.14/2),tr.scale(1.3,1.3,1.5)])
    Aleta1.childs+=[GpuAletas]

    Aleta2 = sg.SceneGraphNode("Aleta2")
    Aleta2.transform=tr.matmul([tr.translate(0,0.6,0.2),tr.rotationZ(-0.5),tr.rotationY(4),tr.rotationX(3.14/2),tr.scale(1.3,1.3,1.5)])
    Aleta2.childs+=[GpuAletas]

    Ojo1 = sg.SceneGraphNode("Ojo1")
    Ojo1.transform=tr.matmul([tr.translate(1,0.53,1),tr.rotationZ(-0.1),tr.rotationX(3.14/2),tr.scale(0.2,0.3,0.2)])
    Ojo1.childs+=[GpuOjo]

    Ojo2 = sg.SceneGraphNode("Ojo2")
    Ojo2.transform=tr.matmul([tr.translate(1,-0.53,1),tr.rotationZ(0.1),tr.rotationX(3.14/2),tr.scale(0.2,0.3,0.2)])
    Ojo2.childs+=[GpuOjo]

    Aletas = sg.SceneGraphNode("Aletas")
    Aletas.childs+=[Aleta1]
    Aletas.childs+=[Aleta2]
    Aletas.childs+=[ColaF]
    Aletas.childs+=[Cresta1]
    Aletas.childs+=[Cresta2]

    Ojos = sg.SceneGraphNode("Ojos")
    Ojos.childs+=[Ojo1]
    Ojos.childs+=[Ojo2]

    todo = sg.SceneGraphNode("todo")
    todo.childs+=[Body]
    todo.childs+=[Aletas]
    todo.childs+=[Ojos]

    return todo


def createPezB():

    GpuCuerpo=esfera=es.toGPUShape(l.esfera(30,30,[225/255,35/255,1/255]))
    GpuOjo=es.toGPUShape(l.generateCirculo(15))
    GpuAletas=es.toGPUShape(createAleta(20))
    GpuAletas2=es.toGPUShape(createAleta2(20))
 
    Cuerpo = sg.SceneGraphNode("Cuerpo")
    Cuerpo.transform=tr.matmul([tr.translate(0,0,0.9),tr.scale(4,0.6,1)])
    Cuerpo.childs+=[GpuCuerpo]
    
    Body = sg.SceneGraphNode("Body")
    Body.childs+=[Cuerpo]

    Cola1 = sg.SceneGraphNode("Cola1")
    Cola1.transform=tr.matmul([tr.rotationY(3.14/2),tr.rotationX(3.14/2),tr.scale(-1,2,1)])
    Cola1.childs+=[GpuAletas2]

    Cola2 = sg.SceneGraphNode("Cola2")
    Cola2.transform=tr.matmul([tr.rotationY(3.14/2),tr.rotationX(3.14/2),tr.scale(1,2,1)])
    Cola2.childs+=[GpuAletas2]

    ColaMov1 = sg.SceneGraphNode("ColaMov1")
    ColaMov1.childs+=[Cola1]

    ColaMov2 = sg.SceneGraphNode("ColaMov2")
    ColaMov2.childs+=[Cola2]

    ColaF1 = sg.SceneGraphNode("ColaF1")
    ColaF1.transform=tr.translate(-4.5,0,1.2)
    ColaF1.childs+=[ColaMov1]

    ColaF2 = sg.SceneGraphNode("ColaF2")
    ColaF2.transform=tr.translate(-4.5,0,0.8)
    ColaF2.childs+=[ColaMov2]

    ColaF = sg.SceneGraphNode("ColaF")
    ColaF.childs+=[ColaF1]
    ColaF.childs+=[ColaF2]

    Cresta1 = sg.SceneGraphNode("Cresta1")
    Cresta1.transform=tr.matmul([tr.translate(-0.5,0,1.8),tr.rotationY(2.3),tr.rotationX(3.14/2),tr.scale(1,1,1)])
    Cresta1.childs+=[GpuAletas]

    Cresta2 = sg.SceneGraphNode("Cresta2")
    Cresta2.transform=tr.matmul([tr.translate(-0.55,0,2.25),tr.rotationY(5.44),tr.rotationX(3.14/2),tr.scale(0.2,1,1)])
    Cresta2.childs+=[GpuAletas]

    Cresta3 = sg.SceneGraphNode("Cresta3")
    Cresta3.transform=tr.matmul([tr.translate(-0.75,0,2.24),tr.rotationY(5.44),tr.rotationX(3.14/2),tr.scale(0.2,1,1)])
    Cresta3.childs+=[GpuAletas]

    Cresta4 = sg.SceneGraphNode("Cresta4")
    Cresta4.transform=tr.matmul([tr.translate(-0.95,0,2.2),tr.rotationY(5.44),tr.rotationX(3.14/2),tr.scale(0.2,1,1)])
    Cresta4.childs+=[GpuAletas]

    Cresta5 = sg.SceneGraphNode("Cresta5")
    Cresta5.transform=tr.matmul([tr.translate(-1.15,0,2.1),tr.rotationY(5.44),tr.rotationX(3.14/2),tr.scale(0.2,1,1)])
    Cresta5.childs+=[GpuAletas]

    Aleta1 = sg.SceneGraphNode("Aleta1")
    Aleta1.transform=tr.matmul([tr.translate(0.5,-0.6,0.3),tr.rotationZ(0.4),tr.rotationY(3.14/2),tr.rotationX(3.14/2),tr.scale(0.7,1.5,1.5)])
    Aleta1.childs+=[GpuAletas]

    Aleta2 = sg.SceneGraphNode("Aleta2")
    Aleta2.transform=tr.matmul([tr.translate(0.5,0.6,0.3),tr.rotationZ(-0.4),tr.rotationY(3.14/2),tr.rotationX(3.14/2),tr.scale(0.7,1.5,1.5)])
    Aleta2.childs+=[GpuAletas]

    Ojo1 = sg.SceneGraphNode("Ojo1")
    Ojo1.transform=tr.matmul([tr.translate(2,0.55,1.2),tr.rotationZ(-0.1),tr.rotationX(3.14/2),tr.scale(0.2,0.3,0.2)])
    Ojo1.childs+=[GpuOjo]

    Ojo2 = sg.SceneGraphNode("Ojo2")
    Ojo2.transform=tr.matmul([tr.translate(2,-0.55,1.2),tr.rotationZ(0.1),tr.rotationX(3.14/2),tr.scale(0.2,0.3,0.2)])
    Ojo2.childs+=[GpuOjo]

    Aletas = sg.SceneGraphNode("Aletas")
    Aletas.childs+=[Aleta1]
    Aletas.childs+=[Aleta2]
    Aletas.childs+=[ColaF]
    Aletas.childs+=[Cresta1]
    Aletas.childs+=[Cresta2]
    Aletas.childs+=[Cresta3]
    Aletas.childs+=[Cresta4]
    Aletas.childs+=[Cresta5]

    Ojos = sg.SceneGraphNode("Ojos")
    Ojos.childs+=[Ojo1]
    Ojos.childs+=[Ojo2]

    todo = sg.SceneGraphNode("todo")
    todo.childs+=[Body]
    todo.childs+=[Aletas]
    todo.childs+=[Ojos]

    return todo

def createPezC():

    GpuCuerpo=esfera=es.toGPUShape(l.esfera(30,30,[1,128/255,0]))
    GpuOjo=es.toGPUShape(l.generateCirculo(15))
    GpuAletas=es.toGPUShape(createAleta(20))
    GpuAletas2=es.toGPUShape(createAleta2(20))

    Cuerpo = sg.SceneGraphNode("Cuerpo")
    Cuerpo.transform=tr.matmul([tr.translate(0,0,0.9),tr.scale(3,0.7,2)])
    Cuerpo.childs+=[GpuCuerpo]

    Body = sg.SceneGraphNode("Body")
    Body.childs+=[Cuerpo]

    Cola1 = sg.SceneGraphNode("Cola1")
    Cola1.transform=tr.matmul([tr.rotationY(3.14/2),tr.rotationX(3.14/2),tr.scale(-2,3,1)])
    Cola1.childs+=[GpuAletas2]

    Cola2 = sg.SceneGraphNode("Cola2")
    Cola2.transform=tr.matmul([tr.rotationY(3.14/2),tr.rotationX(3.14/2),tr.scale(2,3,1)])
    Cola2.childs+=[GpuAletas2]

    ColaMov1 = sg.SceneGraphNode("ColaMov1")
    ColaMov1.childs+=[Cola1]

    ColaMov2 = sg.SceneGraphNode("ColaMov2")
    ColaMov2.childs+=[Cola2]

    ColaF1 = sg.SceneGraphNode("ColaF1")
    ColaF1.transform=tr.translate(-4,0,1.5)
    ColaF1.childs+=[ColaMov1]

    ColaF2 = sg.SceneGraphNode("ColaF2")
    ColaF2.transform=tr.translate(-4,0,0.5)
    ColaF2.childs+=[ColaMov2]

    ColaF = sg.SceneGraphNode("ColaF")
    ColaF.childs+=[ColaF1]
    ColaF.childs+=[ColaF2]

    Cresta1 = sg.SceneGraphNode("Cresta1")
    Cresta1.transform=tr.matmul([tr.translate(-0.5,0,2.3),tr.rotationY(3.14/2),tr.rotationX(3.14/2),tr.scale(4,5,1)])
    Cresta1.childs+=[GpuAletas]

    Cresta2 = sg.SceneGraphNode("Cresta2")
    Cresta2.transform=tr.matmul([tr.translate(-0.5,0,-0.4),tr.rotationY(3.14/2),tr.rotationX(3.14/2),tr.scale(4,5,1)])
    Cresta2.childs+=[GpuAletas]

    Aleta1 = sg.SceneGraphNode("Aleta1")
    Aleta1.transform=tr.matmul([tr.translate(0,-0.75,0.3),tr.rotationZ(0.4),tr.rotationY(3.14/2),tr.rotationX(3.14/2),tr.scale(2,1.3,1.5)])
    Aleta1.childs+=[GpuAletas]

    Aleta2 = sg.SceneGraphNode("Aleta2")
    Aleta2.transform=tr.matmul([tr.translate(0,0.75,0.3),tr.rotationZ(-0.4),tr.rotationY(3.14/2),tr.rotationX(3.14/2),tr.scale(2,1.3,1.5)])
    Aleta2.childs+=[GpuAletas]

    Ojo1 = sg.SceneGraphNode("Ojo1")
    Ojo1.transform=tr.matmul([tr.translate(1.4,0.72,1),tr.rotationZ(-0.1),tr.rotationX(3.14/2),tr.scale(0.2,0.3,0.2)])
    Ojo1.childs+=[GpuOjo]

    Ojo2 = sg.SceneGraphNode("Ojo2")
    Ojo2.transform=tr.matmul([tr.translate(1.4,-0.72,1),tr.rotationZ(0.1),tr.rotationX(3.14/2),tr.scale(0.2,0.3,0.2)])
    Ojo2.childs+=[GpuOjo]

    Aletas = sg.SceneGraphNode("Aletas")
    Aletas.childs+=[Aleta1]
    Aletas.childs+=[Aleta2]
    Aletas.childs+=[ColaF]
    Aletas.childs+=[Cresta1]
    Aletas.childs+=[Cresta2]

    Ojos = sg.SceneGraphNode("Ojos")
    Ojos.childs+=[Ojo1]
    Ojos.childs+=[Ojo2]

    todo = sg.SceneGraphNode("todo")
    todo.childs+=[Body]
    todo.childs+=[Aletas]
    todo.childs+=[Ojos]

    return todo


 

if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 600
    height = 600

    window = glfw.create_window(width, height, "Modelos de peces", None, None)

    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program
    pipeline3D = es.SimpleModelViewProjectionShaderProgram()
    pipeline3DTexture= es.SimpleTextureModelViewProjectionShaderProgram()
    pipelinePhong = ls.SimplePhongShaderProgram()
    pipelinePhongTexture=ls.SimpleTexturePhongShaderProgram()

    # Telling OpenGL to use our shader program

    # Setting up the clear screen color
    glClearColor(0.15, 0.15, 0.15, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Creating shapes on GPU memory

    AXE=es.toGPUShape(bs.createAxis(100))
    gpupez1=createPezA()
    gpupez2=createPezB()
    gpupez3=createPezC()
    import random


    t0 = glfw.get_time()
    camera_theta = np.pi/4
    angulo  =0
    reversa=False
    while not glfw.window_should_close(window):
        # Using GLFW to check for input events
        glfw.poll_events()

        # Getting the time difference from the previous iteration
        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1

        if (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):
            camera_theta -= 2 * dt

        if (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):
            camera_theta += 2* dt

        # Setting up the view transform

        camX = 5 * np.sin(camera_theta)
        camY = 5* np.cos(camera_theta)

        viewPos = np.array([camX, camY, 2])
        #viewPos = np.array([0, 0, 10])

        view = tr.lookAt(
            viewPos,
            np.array([0,0,0]),
            np.array([0,0,1])
        )


        # Setting up the projection transform

        if controller.projection == PROJECTION_ORTHOGRAPHIC:
            projection = tr.ortho(-8, 8, -8, 8, 0.1, 100)

        elif controller.projection == PROJECTION_FRUSTUM:
            projection = tr.frustum(-5, 5, -5, 5, 9, 100)

        elif controller.projection == PROJECTION_PERSPECTIVE:
            projection = tr.perspective(60, float(width)/float(height), 0.1, 100)
        
        else:
            raise Exception()



        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # Drawing shapes with different model transformations



        ###############################################################################

        # pez3=dt/2 y 0.05
        # pez2= dt y 0.1
        # pez1= dt*2 y 0.1
        if   reversa==False:
            angulo+=dt*2
        
            if angulo>0.1:
                reversa=True
              
        elif reversa==True:
            angulo-=dt*2

            if angulo<-0.1:
                reversa=False
                
        




        #########################################################################


        glUseProgram(pipelinePhong.shaderProgram)


        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Kd"), 0.9, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "lightPosition"), -5, -5, 5)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])
        glUniform1ui(glGetUniformLocation(pipelinePhong.shaderProgram, "shininess"), 100)
        
        glUniform1f(glGetUniformLocation(pipelinePhong.shaderProgram, "constantAttenuation"), 0.0001)
        glUniform1f(glGetUniformLocation(pipelinePhong.shaderProgram, "linearAttenuation"), 0.03)
        glUniform1f(glGetUniformLocation(pipelinePhong.shaderProgram, "quadraticAttenuation"), 0.01)

        glUniformMatrix4fv(glGetUniformLocation(pipelinePhong.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipelinePhong.shaderProgram, "view"), 1, GL_TRUE, view)
        

        Cuerpo = sg.findNode(gpupez1, "Body")
        Cuerpo2 = sg.findNode(gpupez2, "Body")
        Cuerpo2.transform= tr.translate(0,3,0)
        Cuerpo3 = sg.findNode(gpupez3, "Body")
        Cuerpo3.transform=tr.translate(0,-3,0)
    
        sg.drawSceneGraphNode(Cuerpo,pipelinePhong, "model")
        sg.drawSceneGraphNode(Cuerpo2,pipelinePhong, "model")
        sg.drawSceneGraphNode(Cuerpo3,pipelinePhong, "model")

        


 
  #############################################################################################################

        glUseProgram(pipelinePhongTexture.shaderProgram)

        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ka"), 0.7, 0.7, 0.7)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Kd"), 0.9, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ks"), 1.0, 1.0, 1.0)


        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "lightPosition"), 0, 0, 10)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])
        glUniform1ui(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "shininess"), 100)
        
        glUniform1f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "constantAttenuation"), 0.01)
        glUniform1f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "linearAttenuation"), 0.03)
        glUniform1f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "quadraticAttenuation"), 0.01)

        glUniformMatrix4fv(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "view"), 1, GL_TRUE, view)

        
        



        #############################################################

        glUseProgram(pipeline3D.shaderProgram)

        glUniformMatrix4fv(glGetUniformLocation(pipeline3D.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3D.shaderProgram, "view"), 1, GL_TRUE, view)
       # glUniformMatrix4fv(glGetUniformLocation(pipeline3D.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
       # pipeline3D.drawShape(AXE,GL_LINES)
        
        
        Aletas1 = sg.findNode(gpupez1, "Aletas")
        Cola1 = sg.findNode(gpupez1, "ColaF")
        Cola1.transform=tr.rotationZ(angulo)

        Aletas2 = sg.findNode(gpupez2, "Aletas")
        Aletas2.transform=tr.translate(0,3,0)
        Cola2 = sg.findNode(gpupez2, "ColaF")
        Cola2.transform=tr.rotationZ(angulo)

        Aletas3 = sg.findNode(gpupez3, "Aletas")
        Aletas3.transform=tr.translate(0,-3,0)
        Cola3 = sg.findNode(gpupez3, "ColaF")
        Cola3.transform=tr.rotationZ(angulo)


        sg.drawSceneGraphNode(Aletas1,pipeline3D, "model")
        sg.drawSceneGraphNode(Aletas2,pipeline3D, "model")
        sg.drawSceneGraphNode(Aletas3,pipeline3D, "model")

        Ojos = sg.findNode(gpupez1, "Ojos")

        Ojos2 = sg.findNode(gpupez2, "Ojos")
        Ojos2.transform=tr.translate(0,3,0)

        Ojos3 = sg.findNode(gpupez3, "Ojos")
        Ojos3.transform=tr.translate(0,-3,0)
            

        sg.drawSceneGraphNode(Ojos,pipeline3D, "model")
        sg.drawSceneGraphNode(Ojos2,pipeline3D, "model")
        sg.drawSceneGraphNode(Ojos3,pipeline3D, "model")
        

        







        glfw.swap_buffers(window)

    glfw.terminate()
