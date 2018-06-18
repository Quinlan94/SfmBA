/*
//
// Created by quinlan on 18-5-8.
//

#ifndef SFM_ARCBALL_H
#define SFM_ARCBALL_H

#endif //SFM_ARCBALL_H

//#include <GL/glut.h>
#include <Eigen/Core>
using namespace std;

#define WIDTH 1024
#define HEIGHT 768

bool mouse_is_pressed = false;
bool mouse_left_is_pressed = false;
bool mouse_right_is_pressed = false;

int prev_mouse_x =0,prev_mouse_y=0,mouse_x=0,mouse_y=0;
float angle;
Eigen::Vector3f axis;

Eigen::Vector3f PositionToArcballVector(
        const float x, const float y)
{
    Eigen::Vector3f vec(2.0f * x / WIDTH - 1, 1 - 2.0f * y / HEIGHT, 0.0f);
    const float norm2 = vec.squaredNorm();
    if (norm2 <= 1.0f) {
        vec.z() = std::sqrt(1.0f - norm2);
    } else {
        vec = vec.normalized();
    }
    return vec;
}
void RotateView(const float x, const float y,
                const float prev_x, const float prev_y)
{
        if (x - prev_x == 0 && y - prev_y == 0) {
            return;
        }
        //轨迹球向量求解
        const Eigen::Vector3f u = PositionToArcballVector(x, y);
        const Eigen::Vector3f v = PositionToArcballVector(prev_x, prev_y);

        // 旋转角度
        angle = 2.0f * std::acos(std::min(1.0f, u.dot(v)))*57.2957795;
        //定义一个最小角度
        const float kMinAngle = 1e-3f;
        if (angle > kMinAngle)
        {
            // 旋转轴
            cout << "旋转。。。。。"<<endl;
            axis = v.cross(u);
            axis = axis.normalized();
            //glRotatef(57.2957795*angle,axis(0),axis(1),axis(2));
        }
    cout<<"x: "<<x<<" y: "<<y<<" prex : "<<prev_x<<" prey: "<<prev_y<<endl;
    cout<< "角度 "<<angle<<"\n"<<" 旋转轴 "<<axis<<endl;
        glutPostRedisplay();



}
void TranslateView(const float x, const float y,
                                      const float prev_x, const float prev_y)
{

}

void mouseMoveEvent(int x,int y)
{
    mouse_x = x;
    mouse_y = y;
    cout << "move !!"<<endl;
    if(mouse_left_is_pressed == true)
    {


        RotateView(mouse_x, mouse_y, prev_mouse_x,
                   prev_mouse_y);
    }
    else if(mouse_right_is_pressed == true)
    {
        TranslateView(x, y, prev_mouse_x,
                      prev_mouse_y);
    }

    prev_mouse_x = mouse_x;
    prev_mouse_y = mouse_y;

}


void mousePressEvent(int button,int state,int x,int y)
{
    if (button == GLUT_LEFT_BUTTON )
    {
        if(state == GLUT_DOWN )
        {
            mouse_left_is_pressed = true;

        }
        else if(state == GLUT_UP)
        {
            mouse_left_is_pressed = false;
        }


    }
    if (button == GLUT_RIGHT_BUTTON )
    {
        if(state == GLUT_DOWN )
        {
            mouse_right_is_pressed = true;

        }
        else if(state == GLUT_UP)
        {
            mouse_right_is_pressed = false;
        }


    }



}
*/
