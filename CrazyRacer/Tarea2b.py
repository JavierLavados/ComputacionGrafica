# coding=utf-8
"""
Daniel Calderon
Universidad de Chile, CC3501, 2018-2
Hermite and Bezier curves using python, numpy and matplotlib
"""

import math
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys

import transformations as tr
import basic_shapes as bs
import easy_shaders as es
import lighting_shaders as l
import local_shapes as ls
import triangle_mesh as tm
import scene_graph as sg



class Controller:
    def __init__(self):
        self.fillPolygon = True
    


# We will use the global controller as communication with the callback function
controller = Controller()


def on_key(window, key, scancode, action, mods):

    if action != glfw.PRESS:
        return
    
    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)



#####################     Modelo de la pista      #####################################################3333

def generateT(t):
    return np.array([[1, t, t**2, t**3]]).T

def hermiteMatrix(P1, P2, T1, T2):
    # Generate a matrix concatenating the columns
    G = np.concatenate((P1, P2, T1, T2), axis=1)
    # Hermite base matrix is a constant
    Mh = np.array([[1, 0, -3, 2], [0, 0, 3, -2], [0, 1, -2, 1], [0, 0, -1, 1]])    
    return np.matmul(G, Mh)

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

def distance(P1,P2): 
    # Saca la distancia entre dos puntos sin importar el Z
    return np.sqrt((P1[0]-P2[0])**2+(P1[1]-P2[1])**2)

def heron(P1,P2,P3): 
    # Saca el area de un triangulo apartir de sus vertices
    a=distance(P1,P2)
    b=distance(P2,P3)
    c=distance(P3,P1)
    s=(a+b+c)/2
    area=np.sqrt( s * (s-a) * (s-b) * (s-c) )
    return area
    
def dentro(P1,P2,P3,P): 
    # Retorna true si un punto P , esta dentro de un triangulo P1,P2 y P3
    t0=heron(P1,P2,P3)
    t1=heron(P1,P2,P)
    t2=heron(P2,P3,P)
    t3=heron(P3,P1,P)
    if t0-0.1<=t1+t2+t3<=t0+0.1:
        return True
    return False

def generatePistaTexture(P1,P2,T1,T2,N,indice,Mode):
    # Genera un segmento de pista usando texturas y se guarda de forma ordenada los vertices y indices para 
    # que al momento de crear la malla de traingulos estos se creen de forma ordenada
    # P1:punto inicial
    # P2:punto final 
    # T1:tangente inicial
    # T2:tangente final 
    # N:cuantos valores entre 0 y 1
    # indice: señala a partir de que indice debe partir los indices
    # Mode: define de como crear la pista

    GMh = hermiteMatrix(P1,P2, T1, T2)
    curve1 = evalCurve(GMh, N)

    vertices = []
    indices = []
    start_index = indice
    
    a=curve1[0]
    b=curve1[1]
    c=curve1[2]

    D1=b-a
    D2=c-b

    p1=Normalizar(np.cross(D1,np.array([0,0,1]))) # derecho inferior
    p2=-p1                                        # izquierdo inferior
    p3=Normalizar(np.cross(D2,np.array([0,0,1]))) # derecho superior
    p4=-p3                                        # izquierdo superior

    v1 = np.array(p2-p1)
    v2 = np.array(a+p1-b-p3)
    v1xv2 = np.cross(v1, v2)
    
    if Mode==True:
        vertices += [a[0]+p1[0],a[1]+p1[1],a[2]+p1[2]  ,1,0,v1xv2[0],v1xv2[1],v1xv2[2],
                   a[0]+p2[0],a[1]+p2[1],a[2]+p2[2]  ,0,0,v1xv2[0],v1xv2[1],v1xv2[2] ]
    if Mode==False:
        vertices += [a[0]+p1[0],a[1]+p1[1],a[2]+p1[2]  ,1,1,v1xv2[0],v1xv2[1],v1xv2[2],
                   a[0]+p2[0],a[1]+p2[1],a[2]+p2[2]  ,0,1,v1xv2[0],v1xv2[1],v1xv2[2] ]
 
    first=Mode
     
    for i in range(curve1.shape[0]-2):
    
        a=curve1[i]
        b=curve1[i+1]
        c=curve1[i+2]

        D1=b-a
        D2=c-b

        p1=Normalizar(np.cross(D1,np.array([0,0,1]))) # derecho inferior
        p2=-p1                            # izquierdo inferior
        p3=Normalizar(np.cross(D2,np.array([0,0,1]))) # derecho superior
        p4=-p3                            #iquierdo superior
        # d === c
        # |     |
        # |     |
        # a === b       
        if first==True:
            _vertex, _indices = ls.createTextureNormalsQuadIndexationFirst(start_index,a+ p2, a+p1, b+p3, b+p4)
            first=False
        else:
            _vertex, _indices = ls.createTextureNormalsQuadIndexationSecond(start_index,a+ p2,a+ p1, b+p3, b+p4)
            first=True

        indices  += _indices
        start_index += 2
        vertices += _vertex[16:len(_vertex) ]

    return (bs.Shape(vertices,indices),start_index,first)

def Normalizar(v):
    # Normaliza el vector v
    modulo = np.sqrt((v[0]**2)+(v[1]**2)+(v[2]**2))
    VectorNormalizado = [v[0]/modulo,v[1]/modulo,v[2]/modulo]

    return np.array(VectorNormalizado)
        
def Escalar(Puntos,velocidad):
    # i: posicion de un punto
    # j: posicion del punto siguiente a i 
    # d: distancia entre el punto Puntos[i] y Puntos[j]
    # a: velocidad escalada de acuerdo a la distancia
    Velocidades = []
    for i in range(len(Puntos)):
        j = (i+1)%len(Puntos)
        P1= Puntos[i]
        P2 = Puntos[j]
        d = np.sqrt( ( P1[0]-P2[0] )**2 + ( P1[1]-P2[1] )**2 + ( P1[2]-P2[2] )**2 )
        a = d*velocidad
        Velocidades.append(a)
    return Velocidades

def Tangentes(Puntos):
    # i: posicion de un punto
    # j: posicion del punto siguiente a i 
    # V1: vector entre  Puntos[i-1] y Puntos[i]
    # V2: vector entre  Puntos[j] y Puntos[i] 
    # Vt: vector tangente a Puntos[i]
    # TangenteN: vector Vt normalizado

    TangentesN = []

    for i in range(len(Puntos)):
        j = (i+1)%len(Puntos)
        V1 = Normalizar(Puntos[i-1]-Puntos[i])
        V2 = Normalizar(Puntos[j]-Puntos[i])
        Vt=V2-V1
        TangenteN = Normalizar(Vt)
        TangentesN.append(TangenteN)
    return TangentesN
    

def PistaDeCarreras(P,N,velocidad):
    # P: puntos de la pista 
    # N: cuantos valores entre 0 y 1
    # Velocidad: velocidad con la que se recorrera la pista

    Pista = []
    T = Tangentes(P)
    S = Escalar(P,velocidad)
    indice = 0
    inicial = True

    for i in range(len(P)):
        
        j = (i+1)%len(P)
        si = S[i]
        P1 = np.array([[P[i][0], P[i][1], P[i][2]]]).T
        P2 = np.array([[P[j][0], P[j][1], P[j][2]]]).T
        T1 = np.array([[T[i][0], T[i][1], T[i][2]]]).T*si
        T2 = np.array([[T[j][0], T[j][1], T[j][2]]]).T*si

        pista,last,Mode = generatePistaTexture(P1,P2,T1,T2,N,indice,inicial)
        inicial = not Mode
        Pista.append(pista)
        indice = last
    
    vertices = []
    indices = []
    for i in range(len(Pista)):
        a = Pista[i]
        vertex = a.vertices
        vertices += vertex
        index = a.indices
        indices += index

    # falta agregar los ultimos indices y puntos a mano

    largoI = len(indices)
    I1 = indices[largoI-2] #el mas grande
    I2 = indices[largoI-3] #el segundo mas grande
    
    largoV = len(vertices)
    a = vertices[largoV-16:largoV-8]    #d---c
    b = vertices[largoV-8:largoV]       #|   |
    c = vertices[0:8]                   #b---a
    d = vertices[8:16]

    # Creo dos puntos nuevos 

    Pd = [(a[0]+c[0])/2, (a[1]+c[1])/2, (a[2]+c[2])/2, 1, 1, a[5], a[6], a[7]]
    Pi = [(b[0]+d[0])/2, (b[1]+d[1])/2, (b[2]+d[2])/2, 0, 1, b[5], b[6], b[7]]

    vertices += Pd
    vertices += Pi
    indices += [  I1, I2,I1+1,I1+1,I1+2,  I1]
    indices += [I1+2,I1+1,  0,   0,    1,I1+2]

    return bs.Shape(vertices,indices)

def create_pista_mesh_Texture(pista):

    ## Creamos los vertices
    mesh_vertices = []

    for i in range(0, len(pista.vertices) - 1, 8):
        mesh_vertices.append((pista.vertices[i], pista.vertices[i + 1], pista.vertices[i + 2]))

    ## Creamos los triangulos
    mesh_triangles = []

    for i in range(0, len(pista.indices) - 1, 3):
        mesh_triangles.append(
            tm.Triangle(pista.indices[i], pista.indices[i + 1], pista.indices[i + 2]))

    ## Creamos la malla con un meshBuilder
    mesh_builder = tm.TriangleFaceMeshBuilder()

    for triangle in mesh_triangles:
        mesh_builder.addTriangle(triangle)

    return mesh_builder, mesh_triangles, mesh_vertices
    
def draw_mesh_Texture(mesh, vertices,imagen):

    shape_indices = []
    # Creamos la lista con indices
    for triangle_mesh in mesh.getTriangleFaceMeshes():
        triangle = triangle_mesh.data
        shape_indices += [triangle.a, triangle.b, triangle.c]
    # Creamos la lista de vertices

    return bs.Shape(vertices, shape_indices,imagen)


######################  Modelo del Automovil ################################3###########################

################## Nuevas Shapes  ######################################
def createTriangulo3D(r, g, b):

    vertices=[
        -0.5, -0.5, 0, r, g, b, 0, 0, -1,   #cara de abajo
        0.5, -0.5,  0, r, g, b, 0, 0, -1, 
        0.5,  0.5,  0, r, g, b, 0, 0, -1,
        -0.5,  0.5, 0, r, g, b, 0, 0, -1,

        0.5,  0.5,  0, r, g, b, -1, 0, 0,  #cara de atras
        0.5,  0.5, 0.5, r, g, b, -1, 0, 0,
        -0.5,  0.5, 0.5, r, g, b, -1, 0, 0,
        -0.5,  0.5, 0, r, g, b, -1, 0, 0,

        -0.5, -0.5, 0, r, g, b, 0, 1, 0,
        -0.5,  0.5, 0, r, g, b, 0, 1, 0,  #cara lateral izquierda
        -0.5,  0.5, 0.5, r, g, b, 0, 1, 0,

        -0.5, -0.5, 0, r, g, b, -0.5,0,0.5,   #cara inclinada
        0.5, -0.5,  0, r, g, b, -0.5,0,0.5,
        0.5,  0.5, 0.5, r, g, b, -0.5,0,0.5,
        -0.5,  0.5, 0.5, r, g, b, -0.5,0,0.5,

        0.5, -0.5,  0, r, g, b, 0, -1, 0,  #cara  lateral derecha
        0.5,  0.5,  0, r, g, b, 0, -1, 0,
        0.5,  0.5, 0.5, r, g, b, 0, -1, 0,
    ]
    indices = [
         0, 1, 2, 2, 3, 0,
         4, 5, 6, 6, 7, 4,
         8,9,10,
         11,12,13,13,14,11,
         15,16,17]

    return bs.Shape(vertices, indices)

def createTextureNormalQuad(image_filename):

    # Defining locations and texture coordinates for each vertex of the shape    
    vertices = [
    #   positions        texture
        -0.5, -0.5, 0.0,  0, 1,   0,0,1,
         0.5, -0.5, 0.0, 1, 1,    0,0,1,
         0.5,  0.5, 0.0, 1, 0,    0,0,1,
        -0.5,  0.5, 0.0,  0, 0,   0,0,1 ]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         0, 1, 2,
         2, 3, 0]

    textureFileName = image_filename

    return bs.Shape(vertices, indices, textureFileName)

def createCar():
    # Crea el automovil

    gpuBlackCube = es.toGPUShape(bs.createColorNormalsCube(0,0,0))
    gpuAsiento = es.toGPUShape(bs.createTextureNormalsCube("Asiento.jpg"),GL_REPEAT,GL_NEAREST)
    gpuCube1 = es.toGPUShape(createTriangulo3D(19/255,81/255,216/255))
    gpuCube2 = es.toGPUShape(bs.createColorNormalsCube(0,0,1))
    gpuCube3 = es.toGPUShape(bs.createColorNormalsCube(19/255,81/255,216/255))
    gpuLogo1=es.toGPUShape(bs.createTextureQuad("LicanRay.png"),GL_REPEAT,GL_NEAREST)
    gpuLogo2=es.toGPUShape(bs.createTextureQuad("BrainDead.png"),GL_REPEAT,GL_NEAREST)
    gpuLogo3=es.toGPUShape(bs.createTextureQuad("16.png"),GL_REPEAT, GL_NEAREST)
    gpuLogo4=es.toGPUShape(bs.createTextureQuad("N7.jpg"),GL_REPEAT, GL_NEAREST)
    gpuLogo5=es.toGPUShape(bs.createTextureQuad("Capeta.png"),GL_REPEAT, GL_NEAREST)
    gpuLLanta=es.toGPUShape(createTextureNormalQuad("Llantas.png"),GL_REPEAT,GL_NEAREST)
    gpuCirculo=es.toGPUShape(ls.generateCirculo(70,"Negro.jpg",1.5),GL_REPEAT,GL_NEAREST)
    gpuCilindro = es.toGPUShape(ls.generateCylinderH(70,"Ruedas.jpg", 1.5,1.5,0),GL_REPEAT,GL_NEAREST)

    ################ Ruedas #############

    Circulo1 = sg.SceneGraphNode("circulo")
    Circulo1.transform=tr.matmul([tr.translate(0,0,0.9),tr.scale(1,1,0.3)])
    Circulo1.childs+=[gpuCirculo]

    Circulo2 = sg.SceneGraphNode("circulo2")
    Circulo2.transform=tr.matmul([tr.translate(0,0,-0.3),tr.scale(1,1,0.3)])
    Circulo2.childs+=[gpuCirculo]

    neumatico = sg.SceneGraphNode("neumatico")
    neumatico.transform=tr.matmul([tr.translate(0,0,-0.3),tr.scale(1,1,0.8)])
    neumatico.childs+=[gpuCilindro]

    Llanta1=sg.SceneGraphNode("llanta1")
    Llanta1.transform=tr.matmul([tr.translate(0,0,-0.32),tr.scale(2.11,2.11,1)])
    Llanta1.childs += [gpuLLanta]

    Llanta2=sg.SceneGraphNode("llanta2")
    Llanta2.transform=tr.matmul([tr.translate(0,0,0.92),tr.scale(2.11,2.11,1)])
    Llanta2.childs += [gpuLLanta]

    wheel = sg.SceneGraphNode("wheel")
    wheel.transform = tr.matmul([tr.rotationX(3.14/2),tr.scale(0.09, 0.09, 0.09)])
    wheel.childs += [Circulo1]
    wheel.childs += [Circulo2]
    wheel.childs += [neumatico]
    wheel.childs += [Llanta1]
    wheel.childs += [Llanta2]

    wheelRotation = sg.SceneGraphNode("wheelRotation")
    wheelRotation.childs += [wheel]

    frontRightWheel = sg.SceneGraphNode("frontRightWheel")
    frontRightWheel.transform = tr.translate(0.35, 0.3, -0.3)
    frontRightWheel.childs += [wheelRotation]
    
    frontLeftWheel = sg.SceneGraphNode("frontLeftWheel")
    frontLeftWheel.transform = tr.translate(0.35, -0.255, -0.3)
    frontLeftWheel.childs += [wheelRotation]

    backRightWheel = sg.SceneGraphNode("backRightWheel")
    backRightWheel.transform = tr.translate(-0.35, 0.3, -0.3)
    backRightWheel.childs += [wheelRotation]
    
    backLeftWheel = sg.SceneGraphNode("backLeftWheel")
    backLeftWheel.transform = tr.translate(-0.35, -0.255, -0.3)
    backLeftWheel.childs += [wheelRotation]
    
    Ruedas = sg.SceneGraphNode("ruedas")
    Ruedas.childs += [frontRightWheel]
    Ruedas.childs += [frontLeftWheel]
    Ruedas.childs += [backRightWheel]
    Ruedas.childs += [backLeftWheel]

    ############Chasis #################

    #trompa
    Cube1 = sg.SceneGraphNode("chasis1")
    Cube1.transform = tr.matmul([tr.translate(0.6,0, -0.3), tr.scale(0.5,0.45,0.5),tr.rotationZ(3.14/2)])
    Cube1.childs += [gpuCube1]
    #aleron delantero
    Cube2 = sg.SceneGraphNode("chasis2")
    Cube2.transform = tr.matmul([tr.translate(0.7,0, -0.32), tr.scale(0.3,0.6,-0.05)])
    Cube2.childs += [gpuBlackCube]
    #cuerpo 
    Cube3 = sg.SceneGraphNode("chasis3")
    Cube3.transform = tr.matmul([tr.translate(0,0, -0.3), tr.scale(0.35,0.7,0.1)])
    Cube3.childs += [gpuBlackCube]
    #costado derecho
    Cube4 = sg.SceneGraphNode("chasis4")
    Cube4.transform = tr.matmul([tr.translate(0,-0.25, -0.25), tr.scale(0.35,0.2,0.5),tr.rotationZ(3.14/2)])
    Cube4.childs += [gpuCube1]
    #costado izquierdo
    Cube5 = sg.SceneGraphNode("chasis5")
    Cube5.transform = tr.matmul([tr.translate(0,0.25, -0.25), tr.scale(0.35,0.2,0.5),tr.rotationZ(3.14/2)])
    Cube5.childs += [gpuCube1]
    #cuerpo2
    Cube6 = sg.SceneGraphNode("chasis6")
    Cube6.transform = tr.matmul([tr.translate(0.25,0,-0.18), tr.scale(0.2,0.45,0.25)])
    Cube6.childs += [gpuCube3]
    #trasero
    Cube7 = sg.SceneGraphNode("chasis7")
    Cube7.transform = tr.matmul([tr.translate(-0.37,0,-0.27), tr.scale(0.4,0.44,0.55),tr.rotationZ(-3.14/2)])
    Cube7.childs += [gpuCube1]
 
    Cube8 = sg.SceneGraphNode("chasis8")
    Cube8.transform = tr.matmul([tr.translate(-0.4,-0.15,-0.1), tr.scale(0.05,0.05,0.3)])
    Cube8.childs += [gpuCube2]
    #soporte izquierdo de aleron trasero
    Cube9 = sg.SceneGraphNode("chasis9")
    Cube9.transform = tr.matmul([tr.translate(-0.4,0.15,-0.1), tr.scale(0.05,0.05,0.3)])
    Cube9.childs += [gpuCube2]
    #aleron
    Cube10 = sg.SceneGraphNode("chasis10")
    Cube10.transform = tr.matmul([tr.rotationY(0.4),tr.translate(-0.4,0,-0.1), tr.scale(0.2,0.6,0.03)])
    Cube10.childs += [gpuBlackCube]
    #parte trasera
    Cube11 = sg.SceneGraphNode("chasis11")
    Cube11.transform = tr.matmul([tr.translate(-0.395,0, -0.325), tr.scale(0.35,0.44,0.1)])
    Cube11.childs += [gpuBlackCube]

    Chasis = sg.SceneGraphNode("chasis")
    Chasis.childs += [Cube1]
    Chasis.childs += [Cube2]
    Chasis.childs += [Cube3]
    Chasis.childs += [Cube4]
    Chasis.childs += [Cube5]
    Chasis.childs += [Cube6]
    Chasis.childs += [Cube7]
    Chasis.childs += [Cube8]
    Chasis.childs += [Cube9]
    Chasis.childs += [Cube10]
    Chasis.childs += [Cube11]


    ######### Sillon ################

    respaldo = sg.SceneGraphNode("respaldo")
    respaldo.transform = tr.matmul([tr.translate(0.1,0,-0.2), tr.scale(0.2,0.3,0.05)])
    respaldo.childs += [gpuAsiento]

    cojin = sg.SceneGraphNode("cojim")
    cojin.transform = tr.matmul([tr.rotationY(-0.4),tr.translate(-0.1,0,-0.05), tr.scale(0.05,0.3,0.3)])
    cojin.childs += [gpuAsiento]

    Asiento = sg.SceneGraphNode("asiento")
    Asiento.childs += [respaldo]
    Asiento.childs += [cojin]

    ########### logos ###########
    cartel1 = sg.SceneGraphNode("cartel")
    cartel1.transform=tr.matmul([tr.translate(0,-0.36,-0.3),tr.scale(0.25,0.15,0.07),tr.rotationX(3.14/2)])
    cartel1.childs += [gpuLogo1]

    cartel2 = sg.SceneGraphNode("cartel2")
    cartel2.transform=tr.matmul([tr.translate(-0.575,0,-0.32),tr.rotationX(3.14/2),tr.rotationY(3.14/2),tr.scale(-0.5,0.1,1)])
    cartel2.childs += [gpuLogo2]

    cartel3 = sg.SceneGraphNode("cartel3")
    cartel3.transform=tr.matmul([tr.translate(0.6,0,-0.17),tr.rotationY(0.45),tr.rotationZ(3.14/2),tr.scale(0.4,0.4,0.4)])
    cartel3.childs += [gpuLogo3]

    cartel4 = sg.SceneGraphNode("cartel4")
    cartel4.transform=tr.matmul([tr.translate(0,0.36,-0.3),tr.scale(-0.2,0.1,0.07),tr.rotationX(3.14/2)])
    cartel4.childs += [gpuLogo4]

    cartel5 = sg.SceneGraphNode("cartel5")
    cartel5.transform=tr.matmul([tr.translate(-0.39,0,0.08),tr.rotationY(0.4),tr.rotationZ(1.35),tr.scale(0.5,0.22,1)])
    cartel5.childs += [gpuLogo5]

    logos = sg.SceneGraphNode("logos")
    logos.childs += [cartel1]
    logos.childs += [cartel2]
    logos.childs += [cartel3]
    logos.childs += [cartel4]
    logos.childs += [cartel5]
 
    car=sg.SceneGraphNode("car")
    car.childs += [Chasis] 
    car.childs += [Ruedas]
    car.childs += [Asiento]
    car.childs += [logos]

    return car

############################# Modelos de las animaciones ############################################

def createLapiz():
    # Geneta el lapiz

    gpuCuerpo=es.toGPUShape(ls.generateCylinderH(8,"Lapiz.jpg",0.5,7,0),GL_REPEAT,GL_NEAREST)
    gpugoma=es.toGPUShape(ls.generateCylinder(50,"Goma.jpg",0.5,1,0),GL_REPEAT,GL_NEAREST)
    gpupunta=es.toGPUShape(ls.generatecono(50,"Punta.jpg",0.5,1.5,0),GL_REPEAT,GL_NEAREST)
    gpumina=es.toGPUShape(ls.generatecono(50,"Negro.jpg",0.25,0.6,0),GL_REPEAT,GL_NEAREST)
    gpulinea=es.toGPUShape(ls.generateCylinderColor(50,1.7,0.06,0,[0,0,0]))

    cuerpo=sg.SceneGraphNode("cuerpo")
    cuerpo.transform=tr.translate(0,0,-2)
    cuerpo.childs += [gpuCuerpo] 

    goma=sg.SceneGraphNode("goma")
    goma.transform=tr.translate(0,0,5)
    goma.childs += [gpugoma]

    punta=sg.SceneGraphNode("punta")
    punta.transform=tr.translate(0,0,-3.5)
    punta.childs += [gpupunta]

    mina=sg.SceneGraphNode("mina")
    mina.transform=tr.translate(0,0,-3.5)
    mina.childs += [gpumina]

    linea=sg.SceneGraphNode("linea")
    linea.transform=tr.translate(0,0,-3)
    linea.childs += [gpulinea]

    lapiz=sg.SceneGraphNode("lapiz")
    lapiz.childs += [cuerpo]  
    lapiz.childs += [goma]
    lapiz.childs += [punta]
    lapiz.childs += [mina]

    dibujo=sg.SceneGraphNode("dibujo")
    dibujo.childs += [linea]  

    todo=sg.SceneGraphNode("todo")
    todo.childs += [lapiz] 
    todo.childs += [dibujo]

    return todo

def createMoneda():
    # Genera la moneda

    gpuCuerpo=es.toGPUShape(ls.generateCylinderColor(30,4,2,0,[239/255,184/255,16/255]))
    tapa=es.toGPUShape(bs.createTextureQuad("Moneda.png"),GL_REPEAT,GL_NEAREST)

    cuerpo=sg.SceneGraphNode("cuerpo")
    cuerpo.transform=tr.matmul([tr.translate(0,0,0),tr.rotationY(3.14/2),tr.scale(0.62,0.62,0.62)])
    cuerpo.childs += [gpuCuerpo] 

    cuerpoP=sg.SceneGraphNode("cuerpoP")
    cuerpoP.childs += [cuerpo]  
    
    tapa1=sg.SceneGraphNode("tapa1")
    tapa1.transform=tr.matmul([tr.translate(0,0,0),tr.rotationY(3.14/2),tr.rotationZ(3.14/2),tr.scale(5,5,5)])
    tapa1.childs+=[tapa]

    tapa2=sg.SceneGraphNode("tapa2")
    tapa2.transform=tr.matmul([tr.translate(1.2,0,0),tr.rotationY(3.14/2),tr.rotationZ(3.14/2),tr.scale(5,5,5)])
    tapa2.childs+=[tapa]

    tapas=sg.SceneGraphNode("tapas")
    tapas.childs += [tapa1]
    tapas.childs += [tapa2]  

    todo=sg.SceneGraphNode("todo")
    todo.childs += [tapas] 
    todo.childs += [cuerpoP]

    return todo

#################     Puntos de la curva   #####################################################
P1=[0 ,0 ,0]
P2=[0,3,0]
P3=[ 2,6, 0]
P4=[ 0, 9, 0]
P5=[ 2, 12, 2]
P6=[ 2, 17, 2]
P7=[ -1, 20,  2]
P8=[ -5, 20,  1]
P9=[ -8, 18, 0 ]
P10=[ -11, 20, 0]
P11=[ -18, 15, 0]
P12=[-22,15,1]
P13=[-22,19,2]
P14=[-18,19,3]
P15=[-10,14,2]
P16=[-10,9,0]
P17=[-17,9,0]
P18=[-17,7,0]
P19=[-10,5,2]
P20=[-4,2,0]
P21=[-4,-3,0]
P22=[0,-3,0]




P=np.array([P1,P2,P3,P4,P5,
           P6,P7,P8,P9,P10,
           P11,P12,P13,P14,P15,
           P16,P17,P18,P19,P20,
           P21,P22])


if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 600
    height = 600

    window = glfw.create_window(width, height, "Crazy Racer", None, None)

    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program

    pipeline3D = es.SimpleModelViewProjectionShaderProgram()
    pipeline3DTexture= es.SimpleTextureModelViewProjectionShaderProgram()
    pipelinePhong = l.SimplePhongShaderProgram()
    pipelinePhongTexture=l.SimpleTexturePhongShaderProgram()

    # Telling OpenGL to use our shader program

    # Setting up the clear screen color
    glClearColor(131/255, 181/255, 221/255, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Creating shapes on GPU memory
    #gpuAxis = es.toGPUShape(bs.createAxis(400))

    pistaFinal=PistaDeCarreras(P,40,1)
    pista_mesh, mesh_triangles, mesh_vertices = create_pista_mesh_Texture(pistaFinal)
    cpuSurface = draw_mesh_Texture(pista_mesh, pistaFinal.vertices,"Suelo.jpg")
    gpupistaTexture = es.toGPUShape(cpuSurface,GL_REPEAT, GL_NEAREST)


    redCarNode = createCar()
    gpulapiz=createLapiz()
    gpuMoneda=createMoneda()
    gpuSuelo = es.toGPUShape(bs.createTextureQuad("Hoja.jpg"),GL_REPEAT,GL_NEAREST)
    gpuGente=es.toGPUShape(bs.createTextureQuad("Gente.png"), GL_REPEAT, GL_NEAREST)
    gpuMeta=es.toGPUShape(bs.createTextureQuad("Meta.jpg"), GL_REPEAT, GL_NEAREST)

    t0 = glfw.get_time()
    camera_theta = 0
    carX = 0
    carY = 0
    carZ = 0
    posV1 = 0  # Pos vertice 1 del triagulo en el cual estoy
    posV2 = 0  # Pos vertice 2 del triagulo en el cual estoy
    posV3 = 0  # Pos vertice 3 del triagulo en el cual estoy



    Dentro=True # True si ya me encuentro dentro de un triagulo  
    Avanzar=True # True si estoy avanzando en la pista
    lista=pista_mesh.getTriangleFaceMeshes()
    Triangulo=lista[0]    # variable que define en que triangulo estoy
    lastD="ab"            # variable que me dice cual fue el ultimo lado de un triangulo que atravece avanzando
    lastR="ca"            # variable que me dice cual fue el ultimo lado de un triangulo que atravece retrocediendo
        





    while not glfw.window_should_close(window):
        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # Using GLFW to check for input events
        glfw.poll_events()

        # Getting the time difference from the previous iteration
        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1

        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)


        if (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):
            camera_theta -= 2 * dt
        if (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):
            camera_theta += 2* dt
        if (glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS):
            carX += (50*dt)*np.sin(camera_theta)
            carY += (50*dt)*np.cos(camera_theta)
            Avanzar=True
        if (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS):
            carX -= (50*dt)*np.sin(camera_theta)
            carY -= (50*dt)*np.cos(camera_theta)
            Avanzar=False

        viewPos = np.array([carX - 25*np.sin(camera_theta), carY - 25*np.cos(camera_theta), 15+carZ])

        view = tr.lookAt(
            viewPos,
            np.array([carX, carY,carZ]),
            np.array([0,0,1])
        )

        projection = tr.frustum(-10, 10, -10, 10, 10, 900)


       ########### Transformaciones ##############################     
        
        PistaTransform=tr.scale(25,25,8)

        pos1=mesh_vertices[Triangulo.data.a] # Pos Vertice 1 del triangulo
        pos2=mesh_vertices[Triangulo.data.b] # Pos Vertice 2 del triangulo
        pos3=mesh_vertices[Triangulo.data.c] # pos Vertice 3 del triangulo

        A=[pos1[0],pos1[1],pos1[2],0]
        B=[pos2[0],pos2[1],pos2[2],0]
        C=[pos3[0],pos3[1],pos3[2],0]

        posV1=tr.matmul([A,PistaTransform]) # Pos Vertice 1 del triangulo escalado
        posV2=tr.matmul([B,PistaTransform]) # Pos Vertice 2 del triangulo escalado
        posV3=tr.matmul([C,PistaTransform]) # Pos Vertice 3 del triangulo escalado

        z_aprox=max(posV1[2],posV2[2],posV3[2]) # Altura entre el maximo de las pos Z de los vertices
        carZ=z_aprox

        if not dentro(posV1,posV2,posV3,[carX,carY,carZ]): 
            # si ya no me encuentro en el triagulo en que estaba
            Dentro= False    
            if Avanzar==True:
                if Triangulo.ab!=None and lastD!="ab" and Dentro ==False:
                    # Si mi triagulo tiene alguien en el lado y no he pasado por ese lado
                    Triangulo=Triangulo.ab
                    lastD="ab"
                    lastR="ca"
                    Dentro=True
                if Triangulo.ca!=None and lastD!="ca" and Dentro== False:
                    # Si mi triagulo tiene alguien en el lado y no he pasado por ese lado
                    Triangulo=Triangulo.ca
                    lastD="ca"
                    lastR="ab"
                    Dentro=True
            if Avanzar==False:
                if Triangulo.ab!=None and lastR!="ab" and Dentro ==False:
                    # Si mi triagulo tiene alguien en el lado y no he pasado por ese lado
                    Triangulo=Triangulo.ab
                    lastR="ab"
                    lastD="ca"
                    Dentro=True
                if Triangulo.ca!=None and lastR!="ca" and Dentro== False:
                    # Si mi triagulo tiene alguien en el lado y no he pasado por ese lado
                    Triangulo=Triangulo.ca
                    lastR="ca"
                    lastD="ab"
                    Dentro=True
         
        # Transfomacion de la inclinacion del auto
        if (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):
            carRotation = tr.rotationZ((np.pi/2)-camera_theta+0.2)
        elif (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):
            carRotation = tr.rotationZ((np.pi/2)-camera_theta-0.2)      
        else:
            carRotation = tr.rotationZ((np.pi/2)-camera_theta)

        redWheelRotationNode = sg.findNode(redCarNode, "wheelRotation")
        
        if (glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS):
            redWheelRotationNode.transform = tr.rotationY(5 * glfw.get_time())    
        elif (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS):
            redWheelRotationNode.transform = tr.rotationY(-5 * glfw.get_time())
        else:
            redWheelRotationNode.transform = tr.identity()

         
        # Transformacion de Auto
        ModelCar= tr.matmul([
            tr.translate(carX,carY,carZ+5),
            carRotation,
            tr.scale(12,14,12)
            ])

        chasis = sg.findNode(redCarNode, "chasis")
        chasis.transform= ModelCar
        ruedas = sg.findNode(redCarNode, "ruedas")
        ruedas.transform= ModelCar
        asiento = sg.findNode(redCarNode, "asiento")
        asiento.transform= ModelCar
        logos = sg.findNode(redCarNode, "logos")
        logos.transform= ModelCar
        
        # Transformacion de Lapiz
        ModelLapiz=tr.matmul([tr.translate (-400,20,45),tr.scale(15,15,15)])

        dibujo = sg.findNode(gpulapiz,"dibujo")
        dibujo.transform=ModelLapiz
        lapiz = sg.findNode(gpulapiz,"lapiz")
        lapiz.transform=tr.matmul([ ModelLapiz,tr.rotationZ(t1*2),tr.rotationX(0.5) ])

        # Transformacion de Moneda
        ModelMoneda=tr.matmul([tr.translate(-100,400,25),tr.rotationZ(2*t1),tr.scale(10,10,10)])

        moneda = sg.findNode(gpuMoneda,"cuerpoP")
        moneda.transform=ModelMoneda
        caras = sg.findNode(gpuMoneda,"tapas")
        caras.transform=ModelMoneda
        
        # Transformacion de suelo

        Suelo1 = tr.matmul([
            tr.translate(0,0,-3),
            tr.uniformScale(500)])
        Suelo2 = tr.matmul([
            tr.translate(0,500,-3),
            tr.uniformScale(500)])
        Suelo3 = tr.matmul([
            tr.translate(-500,0,-3),
            tr.scale(-500,500,500)])
        Suelo4 = tr.matmul([
            tr.translate(-500,500,-3),
            tr.scale(-500,500,500)])

        # Transformacion de Linea de Meta

        transformMeta= tr.matmul([
            tr.translate(0,0,0.2),
            tr.rotationZ(0.09),
            tr.scale(50,20,40)])

        # Transformacion de Gente

        transformG1= tr.matmul([
            tr.translate(40,-200,45),
            tr.rotationX(3.14/2),
            tr.scale(300,100,30)])
        transformG2= tr.matmul([
            tr.translate(-250,-200,45),
            tr.rotationX(3.14/2),
            tr.scale(300,100,30)])
        transformG3= tr.matmul([
            tr.translate(-550,-200,45),
            tr.rotationX(3.14/2),
            tr.scale(300,100,30)])
        transformG4= tr.matmul([
            tr.translate(200,-50,45),
            tr.rotationX(3.14/2),
            tr.rotationY(3.14/2),
            tr.scale(300,100,30)])
        transformG5= tr.matmul([
            tr.translate(200,250,45),
            tr.rotationX(3.14/2),
            tr.rotationY(3.14/2),
            tr.scale(300,100,30)])
        transformG6= tr.matmul([
            tr.translate(200,550,45),
            tr.rotationX(3.14/2),
            tr.rotationY(3.14/2),
            tr.scale(300,100,30)])
        transformG7= tr.matmul([
            tr.translate(40,700,45),
            tr.rotationX(3.14/2),
            tr.scale(300,100,30)])
        transformG8= tr.matmul([
            tr.translate(-250,700,45),
            tr.rotationX(3.14/2),
            tr.scale(300,100,30)])
        transformG9= tr.matmul([
            tr.translate(-550,700,45),
            tr.rotationX(3.14/2),
            tr.scale(300,100,30)])
        transformG10= tr.matmul([
            tr.translate(-700,-50,45),
            tr.rotationX(3.14/2),
            tr.rotationY(3.14/2),
            tr.scale(300,100,30)])
        transformG11= tr.matmul([
            tr.translate(-700,250,45),
            tr.rotationX(3.14/2),
            tr.rotationY(3.14/2),
            tr.scale(300,100,30)])
        transformG12= tr.matmul([
            tr.translate(-700,550,45),
            tr.rotationX(3.14/2),
            tr.rotationY(3.14/2),
            tr.scale(300,100,30)])
                        
        #################### Pineline3D ################################################

        glUseProgram(pipeline3D.shaderProgram)

        glUniformMatrix4fv(glGetUniformLocation(pipeline3D.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3D.shaderProgram, "projection"), 1, GL_TRUE, projection)  

        glUniformMatrix4fv(glGetUniformLocation(pipeline3D.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        sg.drawSceneGraphNode(dibujo,pipeline3D, "model") 
        glUniformMatrix4fv(glGetUniformLocation(pipeline3D.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        sg.drawSceneGraphNode(moneda,pipeline3D, "model")
    
        ############### Pipeline Phong Texture ############################################

        glUseProgram(pipelinePhongTexture.shaderProgram)

        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ls"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ka"), 0.4, 0.4, 0.4)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Kd"), 0.9, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ks"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "lightPosition"), carX, carY, carZ+10)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])
        glUniform1ui(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "shininess"), 10)    
        glUniform1f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "constantAttenuation"), 0.0001)
        glUniform1f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "linearAttenuation"), 0.03)
        glUniform1f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "quadraticAttenuation"), 0.01)
        glUniformMatrix4fv(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "projection"), 1, GL_TRUE, projection)

        glUniformMatrix4fv(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "model"), 1, GL_TRUE, PistaTransform)
        pipelinePhongTexture.drawShape(gpupistaTexture)
        glUniformMatrix4fv(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        sg.drawSceneGraphNode(asiento,pipelinePhongTexture, "model")
        glUniformMatrix4fv(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        sg.drawSceneGraphNode(ruedas,pipelinePhongTexture, "model")


        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ls"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ka"), 0.8, 0.8, 0.8)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Kd"), 0.9, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ks"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "viewPosition"), -400, 20, 60)
        glUniform1ui(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "shininess"), 10)    
        glUniform1f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "constantAttenuation"), 0.0001)
        glUniform1f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "linearAttenuation"), 0.03)
        glUniform1f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "quadraticAttenuation"), 0.01)
        

        glUniformMatrix4fv(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        sg.drawSceneGraphNode(lapiz,pipelinePhongTexture, "model")

        ###################   Pineline Phong  #########################
        glUseProgram(pipelinePhong.shaderProgram)

        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Ls"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Ka"), 0.4, 0.4, 0.4)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Kd"), 0.5, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "Ks"), 0.7, 0.7, 0.7)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "lightPosition"), carX,carY ,carZ+ 10)
        glUniform3f(glGetUniformLocation(pipelinePhong.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])
        glUniform1ui(glGetUniformLocation(pipelinePhong.shaderProgram, "shininess"), 1000)
        glUniform1f(glGetUniformLocation(pipelinePhong.shaderProgram, "constantAttenuation"), 0.0001)
        glUniform1f(glGetUniformLocation(pipelinePhong.shaderProgram, "linearAttenuation"), 0.003)
        glUniform1f(glGetUniformLocation(pipelinePhong.shaderProgram, "quadraticAttenuation"), 0.01)
        glUniformMatrix4fv(glGetUniformLocation(pipelinePhong.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipelinePhong.shaderProgram, "view"), 1, GL_TRUE, view)

        glUniformMatrix4fv(glGetUniformLocation(pipelinePhong.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        sg.drawSceneGraphNode(chasis,pipelinePhong, "model")

        ################# Pipeline 3D Texture ##########################
        glUseProgram(pipeline3DTexture.shaderProgram)

        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE, Suelo1)
        
        pipeline3DTexture.drawShape(gpuSuelo)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE, Suelo2)
        pipeline3DTexture.drawShape(gpuSuelo)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE, Suelo3)
        pipeline3DTexture.drawShape(gpuSuelo)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE, Suelo4)
        pipeline3DTexture.drawShape(gpuSuelo)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        sg.drawSceneGraphNode(logos,pipeline3DTexture, "model")
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        sg.drawSceneGraphNode(caras,pipeline3DTexture, "model")
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE,transformMeta)
        pipeline3DTexture.drawShape(gpuMeta)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE,transformG1)
        pipeline3DTexture.drawShape(gpuGente)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE,transformG2)
        pipeline3DTexture.drawShape(gpuGente)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE,transformG3)
        pipeline3DTexture.drawShape(gpuGente)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE,transformG4)
        pipeline3DTexture.drawShape(gpuGente)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE,transformG5)
        pipeline3DTexture.drawShape(gpuGente)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE,transformG6) 
        pipeline3DTexture.drawShape(gpuGente)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE,transformG7)
        pipeline3DTexture.drawShape(gpuGente)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE,transformG8)
        pipeline3DTexture.drawShape(gpuGente)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE,transformG9) 
        pipeline3DTexture.drawShape(gpuGente)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE,transformG10)
        pipeline3DTexture.drawShape(gpuGente)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE,transformG11)
        pipeline3DTexture.drawShape(gpuGente)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE,transformG12) 
        pipeline3DTexture.drawShape(gpuGente)
        glfw.swap_buffers(window)

    glfw.terminate()
    