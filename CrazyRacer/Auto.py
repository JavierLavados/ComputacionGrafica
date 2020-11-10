
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


def createTriangulo3D(r, g, b):

    # Defining the location and colors of each vertex  of the shape

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


    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         0, 1, 2, 2, 3, 0,
         4, 5, 6, 6, 7, 4,
         8,9,10,
         11,12,13,13,14,11,
         15,16,17]

    return Shape(vertices, indices)


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

    return Shape(vertices, indices, textureFileName)


class Shape:
    def __init__(self, vertices, indices, textureFileName=None):
        self.vertices = vertices
        self.indices = indices
        self.textureFileName = textureFileName





def createCar():

    gpuBlackCube = es.toGPUShape(bs.createColorNormalsCube(0,0,0))
    gpuAsiento = es.toGPUShape(bs.createTextureNormalsCube("Asiento.jpg"),GL_REPEAT,GL_NEAREST)
    gpuCube1 = es.toGPUShape(createTriangulo3D(19/255,81/255,216/255))
    gpuCube2 = es.toGPUShape(bs.createColorNormalsCube(0,0,1))
    gpuCube3 = es.toGPUShape(bs.createColorNormalsCube(19/255,81/255,216/255))
    gpuLogo1=es.toGPUShape(bs.createTextureQuad("LicanRay.png"),GL_REPEAT,GL_NEAREST)
    gpuLogo2=es.toGPUShape(bs.createTextureQuad("BrainDead.png"),GL_REPEAT,GL_NEAREST)
    gpuLogo3=es.toGPUShape(bs.createTextureQuad("16.png"),GL_REPEAT, GL_NEAREST)
    gpuLogo4=es.toGPUShape(bs.createTextureQuad("N7.jpg"),GL_REPEAT, GL_NEAREST)
    gpuLogo5=es.toGPUShape(bs.createTextureQuad("capeta.png"),GL_REPEAT, GL_NEAREST)

    gpuLLanta=es.toGPUShape(createTextureNormalQuad("Llantas.png"),GL_REPEAT,GL_NEAREST)

    gpuCirculo=es.toGPUShape(l.generateCirculo(70,"Negro.jpg",1.5),GL_REPEAT,GL_NEAREST)
    gpuCilindro = es.toGPUShape(l.generateCylinderH(70,"Ruedas.jpg", 1.5,1.5,0),GL_REPEAT,GL_NEAREST)


    ################ Ruedas ###################
    # Cheating a single wheel
    
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

    # Instanciating 2 wheels, for the front and back parts
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

    
    ############Chasis ###############################

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


    ####### Sillon #######################

    #sillon
    respaldo = sg.SceneGraphNode("respaldo")
    respaldo.transform = tr.matmul([tr.translate(0.1,0,-0.2), tr.scale(0.2,0.3,0.05)])
    respaldo.childs += [gpuAsiento]
    #respaldo
    cojin = sg.SceneGraphNode("cojim")
    cojin.transform = tr.matmul([tr.rotationY(-0.4),tr.translate(-0.1,0,-0.05), tr.scale(0.05,0.3,0.3)])
    cojin.childs += [gpuAsiento]
    #soporte derecho de aleron trasero

    Asiento = sg.SceneGraphNode("asiento")
    Asiento.childs += [respaldo]
    Asiento.childs += [cojin]


    ########### logos #######################
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
 
  
    ########### Auto ##########3

    car=sg.SceneGraphNode("car")
    car.childs += [Chasis] 
    car.childs += [Ruedas]
    car.childs += [Asiento]
    car.childs += [logos]

    return car

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



if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 600
    height = 600

    window = glfw.create_window(width, height, "Modelo del Auto", None, None)

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


    
    gpuAxis = es.toGPUShape(bs.createAxis(7))
    redCarNode = createCar()

    t0 = glfw.get_time()
    camera_theta = np.pi/4

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

        camX = 10 * np.sin(camera_theta)
        camY = 10 * np.cos(camera_theta)

        viewPos = np.array([camX, camY, 2])
        3#viewPos = np.array([0, 0, 10])

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

        redWheelRotationNode = sg.findNode(redCarNode, "wheelRotation")
        redWheelRotationNode.transform = tr.rotationY(5 * glfw.get_time())
        
        if (glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS):
            redWheelRotationNode.transform = tr.rotationY(5 * glfw.get_time())
        
        elif (glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS):
            redWheelRotationNode.transform = tr.rotationY(-5 * glfw.get_time())
                
        else:
            redWheelRotationNode.transform = tr.identity()


        ###############################################################################

        model=tr.matmul([tr.scale(5,5,5)])

        chasis = sg.findNode(redCarNode, "chasis")
        chasis.transform= model

        ruedas = sg.findNode(redCarNode, "ruedas")
        ruedas.transform= model

        asiento = sg.findNode(redCarNode, "asiento")
        asiento.transform= model

        logos = sg.findNode(redCarNode, "logos")
        logos.transform= model



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

        glUniformMatrix4fv(glGetUniformLocation(pipelinePhong.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        sg.drawSceneGraphNode(chasis,pipelinePhong, "model")

  #############################################################################################################

        glUseProgram(pipelinePhongTexture.shaderProgram)

        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Kd"), 0.9, 0.5, 0.5)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ks"), 1.0, 1.0, 1.0)


        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "lightPosition"), -5, -5, 5)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])
        glUniform1ui(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "shininess"), 100)
        
        glUniform1f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "constantAttenuation"), 0.0001)
        glUniform1f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "linearAttenuation"), 0.03)
        glUniform1f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "quadraticAttenuation"), 0.01)

        glUniformMatrix4fv(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "view"), 1, GL_TRUE, view)

        glUniformMatrix4fv(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        sg.drawSceneGraphNode(asiento,pipelinePhongTexture, "model")

        glUniformMatrix4fv(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        sg.drawSceneGraphNode(ruedas,pipelinePhongTexture, "model")


        #############################################################

        glUseProgram(pipeline3DTexture.shaderProgram)

        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "view"), 1, GL_TRUE, view)

        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        sg.drawSceneGraphNode(logos,pipeline3DTexture, "model")

        #glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        #pipeline.drawShape(gpuAxis, GL_LINES)

        glfw.swap_buffers(window)

    glfw.terminate()
