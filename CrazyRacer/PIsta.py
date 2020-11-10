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
import test as ts


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

def distance(P1,P2):
    return np.sqrt((P1[0]-P2[0])**2+(P1[1]-P2[1])**2)
def heron(P1,P2,P3):
    a=distance(P1,P2)
    b=distance(P2,P3)
    c=distance(P3,P1)
    s=(a+b+c)/2
    area=np.sqrt( s * (s-a) * (s-b) * (s-c) )

    return area
    
def dentro(P1,P2,P3,P):
    t0=heron(P1,P2,P3)
    t1=heron(P1,P2,P)
    t2=heron(P2,P3,P)
    t3=heron(P3,P1,P)
    if t0-0.1<=t1+t2+t3<=t0+0.1:
        return True
    return False



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




def dist(P1,P2):
    dist=np.sqrt((P1[0]-P2[0])**2+(P1[1]-P2[1])**2+(P1[2]-P2[2])**2)
    return dist


def generatePistaTexture(P1,P2,T1,T2,Pj,N,indice,ini):
    
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
    p2=-p1                            # izquierdo inferior
    p3=Normalizar(np.cross(D2,np.array([0,0,1]))) # derecho superior
    p4=-p3                            #iquierdo superior

    v1 = np.array(p2-p1)
    v2 = np.array(a+p1-b-p3)
    v1xv2 = np.cross(v1, v2)
    #print(p1,p2,p3,p4)
    if ini==True:
        vertices += [a[0]+p1[0],a[1]+p1[1],a[2]+p1[2]  ,1,0,v1xv2[0],v1xv2[1],v1xv2[2],
                   a[0]+p2[0],a[1]+p2[1],a[2]+p2[2]  ,0,0,v1xv2[0],v1xv2[1],v1xv2[2] ]
    if ini==False:
        vertices += [a[0]+p1[0],a[1]+p1[1],a[2]+p1[2]  ,1,1,v1xv2[0],v1xv2[1],v1xv2[2],
                   a[0]+p2[0],a[1]+p2[1],a[2]+p2[2]  ,0,1,v1xv2[0],v1xv2[1],v1xv2[2] ]
 

    first=ini
    # We generate a rectangle for every latitude, 
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


P1=[0 ,0 ,0]
P2=[0,3,0]
P3=[ 2,6, 0]
P4=[ 0, 9, 0]
P5=[ 2, 12, 2]
P6=[ 2, 17, 2]
#P7=[ 0, 21, 0]
P8=[ -1, 20,  2]
P9=[ -5, 20,  1]
P10=[ -8, 18, 0 ]
P11=[ -11, 20, 0]
P12=[ -18, 15, 0]
P13=[-22,15,1]
P14=[-22,19,2]
P15=[-18,19,3]
P16=[-10,14,2]
P17=[-10,9,0]
P18=[-17,9,0]
P19=[-17,7,0]
P20=[-10,5,2]
P22=[-4,2,0]
P23=[-4,-3,0]
P24=[0,-3,0]




P=np.array([P1,P2,P3,P4,P5,
           P6,P8,P9,P10,
           P11,P12,P13,
           P14,P15,
           P16,
           P17,P18,P19,P20,
           P22,P23,P24])




def Normalizar(v):
    modulo = np.sqrt((v[0]**2)+(v[1]**2)+(v[2]**2))
    VectorNormalizado = [v[0]/modulo,v[1]/modulo,v[2]/modulo]
    return np.array(VectorNormalizado)
        
def Escalar(Puntos,velocidad):
    Scale = []
    for i in range(len(Puntos)):
        j = (i+1)%len(Puntos)
        V = Puntos[j]-Puntos[i]
        d = np.sqrt((V[0]**2)+(V[1]**2)+(V[2]**2))
        a=d*velocidad
        Scale.append(a)
    return Scale

def Tangentes(Puntos):
    Tangentes = []
    for i in range(len(Puntos)):
        j = (i+1)%len(Puntos)
        V1 = Normalizar(Puntos[i-1]-Puntos[i])
        V2 = Normalizar(Puntos[j]-Puntos[i])
        Tangente = Normalizar(V2-V1)
        Tangentes.append(Tangente)
    return Tangentes
    
# Te genera los puntos del circuito
def PistaDeCarreras(PI,N,velocidad):
    Pista = []

    TI = Tangentes(PI)
    SI = Escalar(PI,velocidad)

    indice=0
    
    inicial=True

    for i in range(len(PI)):
        
        j = (i+1)%len(PI)
        j2=(i+2)%len(PI)
        si = SI[i]
    
        PI1 = np.array([[PI[i][0], PI[i][1], PI[i][2]]]).T
        PI2 = np.array([[PI[j][0], PI[j][1], PI[j][2]]]).T
        TI1 = np.array([[TI[i][0], TI[i][1], TI[i][2]]]).T*si
        TI2 = np.array([[TI[j][0], TI[j][1], TI[j][2]]]).T*si

        pista,last,ini= generatePistaTexture(PI1,PI2,TI1,TI2,PI[j2],N,indice,inicial)
        inicial=not ini
        Pista.append(pista)
        indice=last
    
    vertices=[]
    indices =[]
    for i in range(len(Pista)):
        a=Pista[i]
        vertex=a.vertices
        vertices+=vertex
        index=a.indices
        indices+=index


    largo=len(indices)
    I1=indices[largo-2] #el mas grande
    I2=indices[largo-3] #el segundo mas grande
  

    largo2=len(vertices)
    a=vertices[largo2-16:largo2-8] #derecha    #d---c
    b=vertices[largo2-8:largo2] # izq          #|   |
    c=vertices[0:8]                            #b---a
    d=vertices[8:16]



    Pd=[(a[0]+c[0])/2,(a[1]+c[1])/2,(a[2]+c[2])/2,1,1,a[5],a[6],a[7]]
    Pi=[(b[0]+d[0])/2,(b[1]+d[1])/2,(b[2]+d[2])/2,0,1,b[5],b[6],b[7]]

    vertices+=Pd
    vertices+=Pi
    indices += [ I1, I2,I1+1,I1+1,I1+2,  I1]
    indices += [I1+2,I1+1,0,0,1,I1+2]

    return bs.Shape(vertices,indices)

def createTextureNormalQuad(image_filename, nx=1, ny=1):

    # Defining locations and texture coordinates for each vertex of the shape    
    vertices = [
    #   positions        texture
        -0.5, -0.5, 0.0,  0, ny,0,0,1,
         0.5, -0.5, 0.0, nx, ny,0,0,1,
         0.5,  0.5, 0.0, nx, 0,0,0,1,
        -0.5,  0.5, 0.0,  0, 0,0,0,1]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         0, 1, 2,
         2, 3, 0]

    textureFileName = image_filename

    return Shape(vertices, indices, textureFileName)


if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 600
    height = 600

    window = glfw.create_window(width, height, "Modelo de la Pista", None, None)

    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program
    pipeline = es.SimpleModelViewProjectionShaderProgram()
    pipeline2 = l.SimpleTexturePhongShaderProgram()
    textureShaderProgram = es.SimpleTextureModelViewProjectionShaderProgram()

    # Telling OpenGL to use our shader program

    # Setting up the clear screen color
    glClearColor(0.85, 0.85, 0.85, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    gpuAxis = es.toGPUShape(bs.createAxis(400))
    
    pistaFinal=PistaDeCarreras(P,50,1)

    pista_mesh, mesh_triangles, mesh_vertices = create_pista_mesh_Texture(pistaFinal)
    cpuSurface = draw_mesh_Texture(pista_mesh, pistaFinal.vertices,"Suelo.jpg")
    
    gpupistaTexture2 = es.toGPUShape(cpuSurface,GL_REPEAT, GL_NEAREST)
    
    gpucubo = es.toGPUShape(bs.createTextureNormalsCube("16.png"),GL_REPEAT,GL_NEAREST)
    gpuTrack = es.toGPUShape(bs.createTextureQuad("Hoja.jpg"),GL_REPEAT,GL_NEAREST)


    t0 = glfw.get_time()
    camera_theta = 0
    carX = 0
    carY = 0
    carZ = 0

    z_previuos=0

    posX = 0
    posY = 0
    posZ = 0



    Dentro=True
    lista=pista_mesh.getTriangleFaceMeshes()
    Triangulo=lista[0]    # variable que define en que triangulo estoy
    last="ab"    # variable que me dice cual fue la ultima variable que utilice entre ab bc y ca
    n=0    

    while not glfw.window_should_close(window):


        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
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
            
        if (glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS):
            carX += (50*dt)*np.sin(camera_theta)
            carY += (50*dt)*np.cos(camera_theta)

        if (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS):
            carX -= (50*dt)*np.sin(camera_theta)
            carY -= (50*dt)*np.cos(camera_theta)

        # Setting up the view transform

        viewPos = np.array([carX - 25*np.sin(camera_theta), carY - 25*np.cos(camera_theta), 15+carZ])

        view = tr.lookAt(
            viewPos,
            np.array([carX, carY,carZ]),
            np.array([0,0,1])
        )

        #projection = tr.perspective(60, float(width)/float(height), 0.1, 100)
        projection = tr.frustum(-10, 10, -10, 10, 10, 400)

        model=tr.scale(15,10,4)
    

        glUseProgram(pipeline.shaderProgram)

        # Setting up the projection transform

        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)



        ##################################################################################
        lightingPipeline=pipeline2

        glUseProgram(lightingPipeline.shaderProgram)


        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Kd"), 0.9, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "lightPosition"), 5+carX, 5+carY, 60)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])
        glUniform1ui(glGetUniformLocation(lightingPipeline.shaderProgram, "shininess"), 10)
        
        glUniform1f(glGetUniformLocation(lightingPipeline.shaderProgram, "constantAttenuation"), 0.0001)
        glUniform1f(glGetUniformLocation(lightingPipeline.shaderProgram, "linearAttenuation"), 0.03)
        glUniform1f(glGetUniformLocation(lightingPipeline.shaderProgram, "quadraticAttenuation"), 0.01)

        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)



        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "model"), 1, GL_TRUE, model)
        lightingPipeline.drawShape(gpupistaTexture2)

    


        #######################################################################################################        
        if (glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS):
            carRotation = tr.rotationZ((np.pi/2)-camera_theta+0.2)
        
        elif (glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS):
            carRotation = tr.rotationZ((np.pi/2)-camera_theta-0.2)
                
        else:
            carRotation = tr.rotationZ((np.pi/2)-camera_theta)
################################################################################################################################3
        lightingPipeline=pipeline2

        glUseProgram(lightingPipeline.shaderProgram)


        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Kd"), 0.9, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "lightPosition"), 5+carX, 5+carY, 15)
        glUniform3f(glGetUniformLocation(lightingPipeline.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])
        glUniform1ui(glGetUniformLocation(lightingPipeline.shaderProgram, "shininess"), 10)
        
        glUniform1f(glGetUniformLocation(lightingPipeline.shaderProgram, "constantAttenuation"), 0.0001)
        glUniform1f(glGetUniformLocation(lightingPipeline.shaderProgram, "linearAttenuation"), 0.03)
        glUniform1f(glGetUniformLocation(lightingPipeline.shaderProgram, "quadraticAttenuation"), 0.01)

        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    
        pos1=mesh_vertices[Triangulo.data.a]
        pos2=mesh_vertices[Triangulo.data.b]
        pos3=mesh_vertices[Triangulo.data.c]
        A=[pos1[0],pos1[1],pos1[2],0]
        B=[pos2[0],pos2[1],pos2[2],0]
        C=[pos3[0],pos3[1],pos3[2],0]
        posX=tr.matmul([A,model])
        posY=tr.matmul([B,model])
        posZ=tr.matmul([C,model])
        z_aprox=max(posX[2],posY[2],posZ[2])
        carZ=z_aprox
        

            
        if not dentro(posX,posY,posZ,[carX,carY,carZ]):

            Dentro= False
            
            
            

            if Triangulo.ab!=None and last!="ab" and Dentro ==False:

                Triangulo=Triangulo.ab
                last="ab"

                Dentro=True
                

            if Triangulo.ca!=None and last!="ca" and Dentro== False:

                Triangulo=Triangulo.ca
                last="ca"

                Dentro=True

        model= tr.matmul([
            tr.translate(carX,carY,carZ+1.7),
            carRotation,
            tr.scale(7,4,4)
            ])

        
        glUniformMatrix4fv(glGetUniformLocation(lightingPipeline.shaderProgram, "model"), 1, GL_TRUE, model )
        lightingPipeline.drawShape(gpucubo)


        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)


        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)

    glfw.terminate()
    