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


def createLapiz():

    gpuCuerpo=es.toGPUShape(l.generateCylinderH(8,"Lapiz.jpg",0.5,7,0),GL_REPEAT,GL_NEAREST)
    gpugoma=es.toGPUShape(l.generateCylinder(50,"Goma.jpg",0.5,1,0),GL_REPEAT,GL_NEAREST)
    gpupunta=es.toGPUShape(l.generatecono(50,"Punta.jpg",0.5,1.5,0),GL_REPEAT,GL_NEAREST)
    gpumina=es.toGPUShape(l.generatecono(50,"Negro.jpg",0.25,0.6,0),GL_REPEAT,GL_NEAREST)
    gpulinea=es.toGPUShape(l.generateCylinderColor(50,1.7,0.06,0,[0,0,0]))

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

    gpuCuerpo=es.toGPUShape(l.generateCylinderColor(30,4,2,0,[239/255,184/255,16/255]))
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

if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 600
    height = 600

    window = glfw.create_window(width, height, "Animaciones", None, None)

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



    gpulapiz=createLapiz()
    gpuMoneda=createMoneda()

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



        ###############################################################################

        model=tr.matmul([tr.scale(1,1,1)])

        lapiz = sg.findNode(gpulapiz,"lapiz")
        lapiz.transform= tr.matmul([tr.rotationZ(t1*2),tr.rotationX(0.5)])




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


  #############################################################################################################

        glUseProgram(pipelinePhongTexture.shaderProgram)

        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "La"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

        glUniform3f(glGetUniformLocation(pipelinePhongTexture.shaderProgram, "Ka"), 0.7, 0.7, 0.7)
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
        sg.drawSceneGraphNode(lapiz,pipelinePhongTexture, "model")


        #############################################################

        glUseProgram(pipeline3D.shaderProgram)

        glUniformMatrix4fv(glGetUniformLocation(pipeline3D.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3D.shaderProgram, "view"), 1, GL_TRUE, view)

        dibujo = sg.findNode(gpulapiz,"dibujo")
        glUniformMatrix4fv(glGetUniformLocation(pipeline3D.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        sg.drawSceneGraphNode(dibujo,pipeline3D, "model") 

        moneda = sg.findNode(gpuMoneda,"cuerpoP")
        moneda.transform=tr.matmul([tr.translate(7,0,0),tr.rotationZ(2*t1)])
        glUniformMatrix4fv(glGetUniformLocation(pipeline3D.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        sg.drawSceneGraphNode(moneda,pipeline3D, "model") 
  

        #########################################################################

        glUseProgram(pipeline3DTexture.shaderProgram)

        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "view"), 1, GL_TRUE, view)


        caras = sg.findNode(gpuMoneda,"tapas")
        caras.transform=tr.matmul([tr.translate(7,0,0),tr.rotationZ(2*t1)])
        glUniformMatrix4fv(glGetUniformLocation(pipeline3DTexture.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
        sg.drawSceneGraphNode(caras,pipeline3DTexture, "model")






        glfw.swap_buffers(window)

    glfw.terminate()
