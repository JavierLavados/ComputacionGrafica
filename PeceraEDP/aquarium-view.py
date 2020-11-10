# coding=utf-8

import numpy as np
import sys
import json
import random 


import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np



import transformations as tr
import basic_shapes as bs
import easy_shaders as es
import scene_graph as sg
import lighting_shaders as ls
import local_shapes as l



#####################################  Peces ################################################################
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





########################## Acuario #########################

def createPecera():

    Gpufierro=es.toGPUShape(bs.createColorCube(0,0,0))
    GpuVidrio=es.toGPUShape(bs.createColorQuad(10/255,80/255,255/255))
    GpuSuelo = es.toGPUShape(bs.createTextureQuad("grava.jpg"),GL_REPEAT,GL_NEAREST)
    Gpuagua=es.toGPUShape(bs.createTextureQuad("suelo.jpg"),GL_REPEAT,GL_NEAREST)

    fierro1=sg.SceneGraphNode("fierro1")
    fierro1.transform=tr.matmul([tr.translate(3,6,4),tr.scale(0.1,0.1,8)])
    fierro1.childs+=[Gpufierro]

    fierro2=sg.SceneGraphNode("fierro2")
    fierro2.transform=tr.matmul([tr.translate(-3,6,4),tr.scale(0.1,0.1,8)])
    fierro2.childs+=[Gpufierro]

    fierro3=sg.SceneGraphNode("fierro3")
    fierro3.transform=tr.matmul([tr.translate(3,-6,4),tr.scale(0.1,0.1,8)])
    fierro3.childs+=[Gpufierro]

    fierro4=sg.SceneGraphNode("fierro4")
    fierro4.transform=tr.matmul([tr.translate(-3,-6,4),tr.scale(0.1,0.1,8)])
    fierro4.childs+=[Gpufierro]

    fierro5=sg.SceneGraphNode("fierro5")
    fierro5.transform=tr.matmul([tr.translate(3,0,8),tr.scale(0.1,12,0.1)])
    fierro5.childs+=[Gpufierro]

    fierro6=sg.SceneGraphNode("fierro6")
    fierro6.transform=tr.matmul([tr.translate(-3,0,8),tr.scale(0.1,12,0.1)])
    fierro6.childs+=[Gpufierro]

    fierro7=sg.SceneGraphNode("fierro7")
    fierro7.transform=tr.matmul([tr.translate(3,0,0),tr.scale(0.1,12,0.1)])
    fierro7.childs+=[Gpufierro]

    fierro8=sg.SceneGraphNode("fierro8")
    fierro8.transform=tr.matmul([tr.translate(-3,0,0),tr.scale(0.1,12,0.1)])
    fierro8.childs+=[Gpufierro]

    fierro9=sg.SceneGraphNode("fierro9")
    fierro9.transform=tr.matmul([tr.translate(0,-6,0),tr.scale(6,0.1,0.1)])
    fierro9.childs+=[Gpufierro]

    fierro10=sg.SceneGraphNode("fierro10")
    fierro10.transform=tr.matmul([tr.translate(0,6,0),tr.scale(6,0.1,0.1)])
    fierro10.childs+=[Gpufierro]

    fierro11=sg.SceneGraphNode("fierro11")
    fierro11.transform=tr.matmul([tr.translate(0,-6,8),tr.scale(6,0.1,0.1)])
    fierro11.childs+=[Gpufierro]

    fierro12=sg.SceneGraphNode("fierro12")
    fierro12.transform=tr.matmul([tr.translate(0,6,8),tr.scale(6,0.1,0.1)])
    fierro12.childs+=[Gpufierro]

    vidrio1=sg.SceneGraphNode("vidrio1")
    vidrio1.transform=tr.matmul([tr.translate(-3,0,4),tr.rotationY(3.14/2),tr.scale(8,12,1)])
    vidrio1.childs+=[GpuVidrio]

    vidrio2=sg.SceneGraphNode("vidrio2")
    vidrio2.transform=tr.matmul([tr.translate(3,0,4),tr.rotationY(3.14/2),tr.scale(8,12,1)])
    vidrio2.childs+=[GpuVidrio]

    vidrio3=sg.SceneGraphNode("vidrio3")
    vidrio3.transform=tr.matmul([tr.translate(0,6,4),tr.rotationX(3.14/2),tr.scale(6,8,1)])
    vidrio3.childs+=[GpuVidrio]

    vidrio4=sg.SceneGraphNode("vidrio4")
    vidrio4.transform=tr.matmul([tr.translate(0,-6,4),tr.rotationX(3.14/2),tr.scale(6,8,1)])
    vidrio4.childs+=[GpuVidrio]

    Suelo=sg.SceneGraphNode("Suelo")
    Suelo.transform=tr.matmul([tr.translate(0,0,0),tr.scale(6,12,1)])
    Suelo.childs+=[GpuSuelo]

    Floor=sg.SceneGraphNode("Floor")
    Floor.childs+=[Suelo]

    Agua=sg.SceneGraphNode("Agua")
    Agua.transform=tr.matmul([tr.translate(0,0,8),tr.scale(6,12,1)])
    Agua.childs+=[Gpuagua]

    Water=sg.SceneGraphNode("Water")
    Water.childs+=[Agua]

    Marco = sg.SceneGraphNode("Marco")
    Marco.childs+=[fierro1]
    Marco.childs+=[fierro2]
    Marco.childs+=[fierro3]
    Marco.childs+=[fierro4]
    Marco.childs+=[fierro5]
    Marco.childs+=[fierro6]
    Marco.childs+=[fierro7]
    Marco.childs+=[fierro8]
    Marco.childs+=[fierro9]
    Marco.childs+=[fierro10]
    Marco.childs+=[fierro11]
    Marco.childs+=[fierro12]
    
    Vidrio = sg.SceneGraphNode("Vidrio")
    Vidrio.childs+=[vidrio1]
    Vidrio.childs+=[vidrio2]
    Vidrio.childs+=[vidrio3]
    Vidrio.childs+=[vidrio4]

    Todo = sg.SceneGraphNode("Todo")
    Todo.childs+=[Vidrio]
    Todo.childs+=[Marco]
    Todo.childs+=[Floor]
    Todo.childs+=[Water]

    return Todo

################## Voxeles #################################

def createColorCube(i, j, k, X, Y, Z,c):
    l_x = X[i, j, k]
    r_x = X[i+1, j, k]
    b_y = Y[i, j, k]
    f_y = Y[i, j+1, k]
    b_z = Z[i, j, k]
    t_z = Z[i, j, k+1]
    #   positions    colors
    vertices = [
    # Z+: number 1
        l_x, b_y,  t_z, c[0],c[1],c[2],
         r_x, b_y,  t_z, c[0],c[1],c[2],
         r_x,  f_y,  t_z, c[0],c[1],c[2],
        l_x,  f_y,  t_z, c[0],c[1],c[2],
    # Z-: number 6
        l_x, b_y, b_z, c[0],c[1],c[2],
         r_x, b_y, b_z, c[0],c[1],c[2],
         r_x,  f_y, b_z, c[0],c[1],c[2],
        l_x,  f_y, b_z, c[0],c[1],c[2],
    # X+: number 5
         r_x, b_y, b_z, c[0],c[1],c[2],
         r_x,  f_y, b_z, c[0],c[1],c[2],
         r_x,  f_y,  t_z, c[0],c[1],c[2],
         r_x, b_y,  t_z, c[0],c[1],c[2],
    # X-: number 2
        l_x, b_y, b_z, c[0],c[1],c[2],
        l_x,  f_y, b_z, c[0],c[1],c[2],
        l_x,  f_y,  t_z, c[0],c[1],c[2],
        l_x, b_y,  t_z, c[0],c[1],c[2],
    # Y+: number 4
        l_x,  f_y, b_z, c[0],c[1],c[2],
        r_x,  f_y, b_z, c[0],c[1],c[2],
        r_x,  f_y, t_z, c[0],c[1],c[2],
        l_x,  f_y, t_z, c[0],c[1],c[2],
    # Y-: number 3
        l_x, b_y, b_z, c[0],c[1],c[2],
        r_x, b_y, b_z, c[0],c[1],c[2],
        r_x, b_y, t_z, c[0],c[1],c[2],
        l_x, b_y, t_z, c[0],c[1],c[2],
        ]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        4, 5, 1, 1, 0, 4,
        6, 7, 3, 3, 2, 6,
        5, 6, 2, 2, 1, 5,
        7, 4, 0, 0, 3, 7]

    return bs.Shape(vertices, indices)

def merge(destinationShape, strideSize, sourceShape):

    # current vertices are an offset for indices refering to vertices of the new shape
    offset = len(destinationShape.vertices)
    destinationShape.vertices += sourceShape.vertices
    destinationShape.indices += [(offset/strideSize) + index for index in sourceShape.indices]


################## Controller ##############################
# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.TipoA=False # True si se quiere ver los voxeles de la zona A
        self.TipoB=False # True si se quiere ver los voxeles de la zona A
        self.TipoC=False # True si se quiere ver los voxeles de la zona A
        self.on=True     # True si se quiere ver los vidrios y la superficioe de la pecera
        self.Down=False  # True si se quiere agachar o ver desde mas bajo
        
# We will use the global controller as communication with the callback function
controller = Controller()


def on_key(window, key, scancode, action, mods):

    if action != glfw.PRESS:
        return
    
    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_A:
        controller.TipoA=not controller.TipoA

    elif key == glfw.KEY_B:
        controller.TipoB=not controller.TipoB

    elif key == glfw.KEY_C:
        controller.TipoC=not controller.TipoC

    elif key == glfw.KEY_ENTER:
        controller.on=not controller.on

    elif key == glfw.KEY_LEFT_CONTROL:
        controller.Down=not controller.Down

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)

archivo = sys.argv[1]

Data=None
with open(archivo) as file:
    data = json.load(file)
    Data=data   




if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 600
    height = 600

    window = glfw.create_window(width, height, "Aquarium-view", None, None)

    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program
    pipeline = es.SimpleModelViewProjectionShaderProgram()
    pipeline3DTexture= es.SimpleTextureModelViewProjectionShaderProgram()
    pipelinePhong = ls.SimplePhongShaderProgram()
    pipelinePhongTexture=ls.SimpleTexturePhongShaderProgram()

    # Telling OpenGL to use our shader program
    glUseProgram(pipeline.shaderProgram)

    # Setting up the clear screen color
    #glClearColor(81/255,209/255,246/255, 1.0)
    glClearColor(0,1,1, 1.0)
    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    
    # Creating shapes on GPU memory
    gpuAxis = es.toGPUShape(bs.createAxis(7))

    name=Data["filename"] # Obtenfo el nombre del archivo

    ub=np.load(name) # Obtengo la matriz

    load_voxels = ub

    W=ub.shape[0] # Obtengo el ancho
    L=ub.shape[1] # Obtengo el largo
    H=ub.shape[2] # Obtengo la altura
    
    Y,X,Z=np.meshgrid(np.linspace(0,L,L),np.linspace(0,W,W),np.linspace(0,H,H))

    isosurfaceA = bs.Shape([], []) # Shape que contendra los voxeles de la zona A
    isosurfaceB = bs.Shape([], []) # Shape que contendra los voxeles de la zona B
    isosurfaceC = bs.Shape([], []) # Shape que contendra los voxeles de la zona C


    ta = Data["t_a"]
    tb = Data["t_b"]
    tc = Data["t_c"]

    # Now let's draw voxels!
    for i in range(X.shape[0]-1):
        for j in range(X.shape[1]-1):
            for k in range(X.shape[2]-1):
               
                
                if ub[i,j,k]>=(ta-2) and ub[i,j,k]<=(ta+2):
                    temp_shape = createColorCube(i,j,k, X,Y, Z,[0,1,0])
                    merge(destinationShape=isosurfaceA, strideSize=6, sourceShape=temp_shape)

                elif ub[i,j,k]>=(tb-2) and ub[i,j,k]<=(tb+2):
                    temp_shape = createColorCube(i,j,k, X,Y, Z,[1,0,0])
                    merge(destinationShape=isosurfaceB, strideSize=6, sourceShape=temp_shape)

                elif ub[i,j,k]>=(tc-2) and ub[i,j,k]<=(tc+2):
                    temp_shape = createColorCube(i,j,k, X,Y, Z,[1,128/255,0])
                    merge(destinationShape=isosurfaceC, strideSize=6, sourceShape=temp_shape)
                


                
    gpu_surfaceA = es.toGPUShape(isosurfaceA)
    gpu_surfaceB = es.toGPUShape(isosurfaceB)
    gpu_surfaceC = es.toGPUShape(isosurfaceC)

    gpueje=es.toGPUShape(bs.createAxis(20))
    GpuPecera=createPecera()

    Gpuimagen1=es.toGPUShape(bs.createTextureQuad("imagen1.jpg"),GL_REPEAT,GL_NEAREST)
    Gpuimagen2=es.toGPUShape(bs.createTextureQuad("imagen2.jpg"),GL_REPEAT,GL_NEAREST)
    Gpunemo=es.toGPUShape(bs.createTextureQuad("nemo.png"),GL_REPEAT,GL_NEAREST)
    GpuSuelo=es.toGPUShape(bs.createTextureQuad("arena.jpg"),GL_REPEAT,GL_NEAREST)
    GpuMesa=es.toGPUShape(bs.createTextureCube("madera4.jpg"),GL_REPEAT,GL_NEAREST)

    t0 = glfw.get_time()
    camera_theta = np.pi/4
    Zoom=1
    

    ###### GpuPeces #####
    PezA=False # Sera verdadero en caso de que haya areas para el pez de tipo A
    PezB=False # Sera verdadero en caso de que haya areas para el pez de tipo B
    PezC=False # Sera verdadero en caso de que haya areas para el pez de tipo C

    #### Peces A ####

    nA= Data["n_a"]   # Numero de peces de tipo A
    ZonaA=isosurfaceA.vertices # Vertices de la zona A
    UbicacionesA=[]   # Posiciones x,y,z de todos los vertices de la zona A
    PosicionesA=[]    # Posiciones x,y,z de los nA peces
    PecesA=[]         # Los na peces
    VelA=[]           # Velocidades de la cola de los na peces
    OrientacionA=[]   # Orientacion de los na peces

    if len(ZonaA)!=0: # Es para verificar que exista una zona donde pueda colocar los Peces A
        PezA=True
        for i in range(0,len(ZonaA),6):# Obtengo todas las cordenadas de la zona A
            x,y,z=(ZonaA[i],ZonaA[i+1],ZonaA[i+2])
            #if x!=0 and x<W-1 and y!=0 and y<L-1 and z!=0 and z<H-1 :
            UbicacionesA.append((x,y,z))

        for i in range(nA): # Creo todos los peces A
            pez=createPezA() # Creo al pez i
            PecesA.append(pez) # Guardo el pez i
            PosicionesA.append(random.choice(UbicacionesA)) # Escojo una posicion al azar para el pez
            VelA.append(random.randint(5, 30)) # Escojo una velocidad al azar 
            OrientacionA.append(random.randint(0, 6)) # Escojo una orientacion al azar

    ##### Peces B ######

    nB= Data["n_b"]   # Numero de peces de tipo B
    ZonaB=isosurfaceB.vertices # Vertices de la zona B
    UbicacionesB=[]   # Posiciones x,y,z de todos los vertices de la zona B
    PosicionesB=[]    # Posiciones x,y,z de los nb peces
    PecesB=[]         # Los nb peces
    VelB=[]           # Velocidades de la cola de los nb peces
    OrientacionB=[]   # Orientacion de los nb peces


    if len(ZonaB)!=0:# Es para verificar que exista una zona donde pueda colocar los Peces B
        PezB=True
        for i in range(0,len(ZonaB),6):# Obtengo todas las cordenadas de la zona A
            x,y,z=(ZonaB[i],ZonaB[i+1],ZonaB[i+2])
            #if x!=0 and x<W-1 and y!=0 and y<L-1 and z!=0 and z<H-1 :
            UbicacionesB.append((x,y,z))

        for i in range(nB): # Creo todos los peces B
            pez=createPezB() # Creo al pez i
            PecesB.append(pez) # Guardo el pez i
            PosicionesB.append(random.choice(UbicacionesB)) # Escojo una posicion al azar para el pez
            VelB.append(random.randint(5, 30)) # Escojo una velocidad al azar 
            OrientacionB.append(random.randint(0, 6)) # Escojo una orientacion al azar

    ##### Peces c ####

    nC= Data["n_c"]   # Numero de peces de tipo C
    ZonaC=isosurfaceC.vertices # Vertices de la zona C
    UbicacionesC=[]   # Posiciones x,y,z de todos los vertices de la zona C
    PosicionesC=[]    # Posiciones x,y,z de los nb peces
    PecesC=[]         # Los nc peces
    VelC=[]           # Velocidades de la cola de los nc peces
    OrientacionC=[]   # Orientacion de los nc peces

    if len(ZonaC)!=0: # Es para verificar que exista una zona donde pueda colocar los Peces c
        PezC=True
        for i in range(0,len(ZonaC),6):# Obtengo todas las cordenadas de la zona C
            x,y,z=(ZonaC[i],ZonaC[i+1],ZonaC[i+2])
            #if x!=0 and x<W-1 and y!=0 and y<L-1 and z!=0 and z<H-1 :
            UbicacionesC.append((x,y,z))

        for i in range(nC): # Creo todos los peces C
            pez=createPezC() # Creo al pez i
            PecesC.append(pez) # Guardo el pez i
            PosicionesC.append(random.choice(UbicacionesC)) # Escojo una posicion al azar para el pez
            VelC.append(random.randint(5, 30)) # Escojo una velocidad al azar 
            OrientacionC.append(random.randint(0, 6)) # Escojo una orientacion al azar

    while not glfw.window_should_close(window):

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # Using GLFW to check for input events
        glfw.poll_events()

        # Getting the time difference from the previous iteration
        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1

        if (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):
            camera_theta += 2 * dt

        if (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):
            camera_theta -= 2* dt

        if (glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS):
            Zoom -= 2 * dt

        if (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS):
            Zoom += 2* dt

        # Setting up the view transform

        camX = 30 * Zoom*np.sin(camera_theta)
        camY = 30 * Zoom*np.cos(camera_theta)

        if controller.Down:
            camZ=5
        else:
            camZ=12
        
        viewPos = np.array([camX, camY, camZ ])

        view = tr.lookAt(
            viewPos,
            np.array([0,0,4]),
            np.array([0,0,1])
        )

        glUseProgram(pipeline.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        projection = tr.perspective(60, float(width)/float(height), 0.1, 200)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        #glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        #pipeline.drawShape(gpueje,GL_LINES)

        # Aca dibujos los shapes de los voxeles de cada pez cuando estos existan y el usuario lo solicite

        TransformVoxeles=tr.matmul([tr.translate(-5,-9.5,0),tr.scale(10/W,19/L,11/H)])

        if controller.TipoA==True and PezA==True:
            glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, TransformVoxeles)
            pipeline.drawShape(gpu_surfaceA,GL_LINES)

        if controller.TipoB==True and PezB==True:
            glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, TransformVoxeles)
            pipeline.drawShape(gpu_surfaceB,GL_LINES)

        if controller.TipoC==True and PezC==True:
            glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, TransformVoxeles)
            pipeline.drawShape(gpu_surfaceC,GL_LINES)


        # Dibujo el marco del acuario
        fierros = sg.findNode(GpuPecera, "Marco")
        fierros.transform=tr.scale(10/6+0.01,19/12+0.01,11.5/8)
        sg.drawSceneGraphNode(fierros,pipeline, "model")

        # Dibujo las aletas y las colas de los distintos tipos de peces 
        if PezA==True:
            for i in range(nA):
                Aletas= sg.findNode(PecesA[i], "Aletas")
                Cola = sg.findNode(PecesA[i], "ColaF")
                Ojos = sg.findNode(PecesA[i], "Ojos")
                ang=VelA[i]
                Cola.transform=tr.rotationZ(np.sin(t1*ang)/6)
                pos=PosicionesA[i]
                ori=OrientacionA[i]
                Ojos.transform=tr.matmul([tr.translate(pos[0]*(8/W)-4,pos[1]*(17/L)-8.5,pos[2]*(10/H)),tr.scale(0.3,0.3,0.3),tr.rotationZ(ori)])
                Aletas.transform=tr.matmul([tr.translate(pos[0]*(8/W)-4,pos[1]*(17/L)-8.5,pos[2]*(10/H)),tr.scale(0.3,0.3,0.3),tr.rotationZ(ori)])
                sg.drawSceneGraphNode(Aletas,pipeline, "model")
                sg.drawSceneGraphNode(Ojos,pipeline, "model")


        if PezB==True:
            for i in range(nB):
                Aletas= sg.findNode(PecesB[i], "Aletas")
                Cola = sg.findNode(PecesB[i], "ColaF")
                Ojos = sg.findNode(PecesB[i], "Ojos")
                ang=VelB[i]
                ori=OrientacionB[i]
                Cola.transform=tr.rotationZ(np.sin(t1*ang)/6)
                pos=PosicionesB[i]
                Ojos.transform=tr.matmul([tr.translate(pos[0]*(8/W)-4,pos[1]*(17/L)-8.5,pos[2]*(10/H)),tr.scale(0.3,0.3,0.3),tr.rotationZ(ori)])
                Aletas.transform=tr.matmul([tr.translate(pos[0]*(8/W)-4,pos[1]*(17/L)-8.5,pos[2]*(10/H)),tr.scale(0.3,0.3,0.3),tr.rotationZ(ori)])
                sg.drawSceneGraphNode(Aletas,pipeline, "model")
                sg.drawSceneGraphNode(Ojos,pipeline, "model")

        if PezC==True:
            for i in range(nC):
                Aletas= sg.findNode(PecesC[i], "Aletas")
                Cola = sg.findNode(PecesC[i], "ColaF")
                Ojos = sg.findNode(PecesC[i], "Ojos")
                ang=VelC[i]
                ori = OrientacionC[i]
                Cola.transform=tr.rotationZ(np.sin(t1*ang)/6)
                pos=PosicionesC[i]
                Ojos.transform=tr.matmul([tr.translate(pos[0]*(8/W)-4,pos[1]*(17/L)-8.5,pos[2]*(10/H)),tr.scale(0.3,0.3,0.3),tr.rotationZ(ori)])
                Aletas.transform=tr.matmul([tr.translate(pos[0]*(8/W)-4,pos[1]*(17/L)-8.5,pos[2]*(10/H)),tr.scale(0.3,0.3,0.3),tr.rotationZ(ori)])
                sg.drawSceneGraphNode(Aletas,pipeline, "model")
                sg.drawSceneGraphNode(Ojos,pipeline, "model")

        glUseProgram(pipeline3DTexture.shaderProgram)

        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "projection"), 1, GL_TRUE, projection)

        # Dibujo el suelo de la pecera
        Suelo = sg.findNode(GpuPecera, "Floor")
        Suelo.transform=tr.scale(10/6+0.01,19/12+0.01,11/8)
        sg.drawSceneGraphNode(Suelo,pipeline3DTexture, "model")

        #Dibujo las pareces que decorarn el acuario , el suelo y la mesa

        transform=tr.matmul([tr.translate(0,80,28),tr.rotationX(3.14/2),tr.scale(160,70,50)])
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE, transform)
        pipeline3DTexture.drawShape(Gpuimagen1)

        transform=tr.matmul([tr.translate(80,0,28),tr.rotationZ(3.14/2),tr.rotationX(3.14/2),tr.scale(160,70,50)])
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE, transform)
        pipeline3DTexture.drawShape(Gpuimagen2)

        transform=tr.matmul([tr.translate(0,-80,28),tr.rotationX(3.14/2),tr.scale(160,70,50)])
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE, transform)
        pipeline3DTexture.drawShape(Gpuimagen1)

        transform=tr.matmul([tr.translate(-80,0,28),tr.rotationZ(3.14/2),tr.rotationX(3.14/2),tr.scale(160,70,50)])
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE, transform)
        pipeline3DTexture.drawShape(Gpuimagen2)

        transform=tr.matmul([tr.translate(-79.9,0,28),tr.rotationZ(3.14/2),tr.rotationX(3.14/2),tr.scale(50,50,50)])
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE, transform)
        pipeline3DTexture.drawShape(Gpunemo)

        transform=tr.matmul([tr.translate(0,0,-6),tr.scale(150,150,20)])
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE, transform)
        pipeline3DTexture.drawShape(GpuSuelo)

        transform=tr.matmul([tr.translate(0,0,-3.1),tr.scale(12,20,6)])
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE, transform)
        pipeline3DTexture.drawShape(GpuMesa)

        glUseProgram(pipelinePhong.shaderProgram)

        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Ka"), 0.5, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Kd"), 0.5, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Ks"), 0.6, 0.6, 0.6)

        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "lightPosition"), 0, 0, 12)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])
        glUniform1ui(glGetUniformLocation(pipelinePhong.shaderProgram, "shininess"), 100)
        
        glUniform1f(glGetUniformLocation(pipelinePhong.shaderProgram, "constantAttenuation"), 0.01)
        glUniform1f(glGetUniformLocation(pipelinePhong.shaderProgram, "linearAttenuation"), 0.0003)
        glUniform1f(glGetUniformLocation(pipelinePhong.shaderProgram, "quadraticAttenuation"), 0.01)

        glUniformMatrix4fv(glGetUniformLocation(pipelinePhong.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipelinePhong.shaderProgram, "view"), 1, GL_TRUE, view)
   
        # Dubujo los cuerpos de los distintos peces

        if PezA==True:
            for i in range(nA):
                Cuerpo = sg.findNode(PecesA[i], "Body")
                pos=PosicionesA[i]
                ori=OrientacionA[i]
                Cuerpo.transform=tr.matmul([tr.translate(pos[0]*(8/W)-4,pos[1]*(17/L)-8.5,pos[2]*(10/H)),tr.scale(0.3,0.3,0.3),tr.rotationZ(ori)])
                sg.drawSceneGraphNode(Cuerpo,pipelinePhong, "model")
        
        if PezB==True:
            for i in range(nB):
                Cuerpo = sg.findNode(PecesB[i], "Body")
                pos=PosicionesB[i]
                ori=OrientacionB[i]
                Cuerpo.transform=tr.matmul([tr.translate(pos[0]*(8/W)-4,pos[1]*(17/L)-8.5,pos[2]*(10/H)),tr.scale(0.3,0.3,0.3),tr.rotationZ(ori)])
                sg.drawSceneGraphNode(Cuerpo,pipelinePhong, "model")

        if PezC==True:
            for i in range(nC):
                Cuerpo = sg.findNode(PecesC[i], "Body")
                pos=PosicionesC[i]
                ori=OrientacionC[i]
                Cuerpo.transform=tr.matmul([tr.translate(pos[0]*(8/W)-4,pos[1]*(17/L)-8.5,pos[2]*(10/H)),tr.scale(0.3,0.3,0.3),tr.rotationZ(ori)])
                sg.drawSceneGraphNode(Cuerpo,pipelinePhong, "model")


        if controller.on==True:
            glBlendFunc(GL_ONE, GL_ONE)
            #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glUseProgram(pipeline3DTexture.shaderProgram)
            glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "view"), 1, GL_TRUE, view)
            glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "projection"), 1, GL_TRUE, projection)
            
            # Dibujo la superficie del agua
            Agua = sg.findNode(GpuPecera, "Water") 
            Agua.transform=tr.matmul([tr.translate(0,0,0.1),tr.scale(10/6+0.01,19/12+0.01,11/8)])
            sg.drawSceneGraphNode(Agua,pipeline3DTexture, "model")
            
            # Dibujo los vidrios de la pecera
            glUseProgram(pipeline.shaderProgram)
            glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
            glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
            Vidrio = sg.findNode(GpuPecera, "Vidrio")
            Vidrio.transform=tr.scale(10/6+0.01,19/12+0.01,11.5/8)
            sg.drawSceneGraphNode(Vidrio,pipeline, "model")

        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)

    glfw.terminate()
