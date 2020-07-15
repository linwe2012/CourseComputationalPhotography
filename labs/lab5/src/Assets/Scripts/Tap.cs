using System.Collections;
using System.Collections.Generic;
using UnityEngine;




public class Tap : MonoBehaviour
{
    GameObject soldier;
    Vector3 soldier_scale;
    float scale_begin = 2;
    float scale_end = 1;
    float scale_cur = 1;
    bool firstTouch = false;
    float touchTime = 0;
    float xSpeed = 50f;
    float lastX = 0f;
    

    public AnimationCurve scale_curve;

    void Swap<T>(ref T a, ref T b)
    {
        T c = a;
        a = b;
        b = c;
    }


    Vector3 DupVec(Vector3 v)
    {
        return new Vector3(
            v.x, v.y, v.z
            );
    }

    // Start is called before the first frame update
    void Start()
    {
        soldier = GameObject.Find("ImageTarget/Soldier_demo");
        soldier_scale = DupVec(soldier.transform.localScale);
    }


    // void isEnlarge(Vector2 oP1, Vector)

    // Update is called once per frame
    void Update()
    {
        if(scale_cur < 1f)
        {
            // 确定采样点
            scale_cur += 0.04f;
            // 采样
            float sample = scale_curve.Evaluate(scale_cur);
            // 根据采样反应到我们实际需要的scale 大小
            float cur = scale_begin + (scale_end - scale_begin) * sample;
            // 最终改变模型的 Scale
            soldier.transform.localScale = soldier_scale * cur;
        }
        if(Input.GetMouseButton(0))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            RaycastHit hitInfo;
            if(Physics.Raycast(ray, out hitInfo))
            {
                // 是否是单击而且是刚刚点击屏幕
                
                if(Input.touchCount == 1)
                {
                    Touch touch = Input.GetTouch(0);

                    if (touch.phase == TouchPhase.Began)
                    {
                        firstTouch = true;
                        touchTime = Time.time;
                        lastX = Input.mousePosition.x;
                    }

                    // 长按
                    else if (touch.phase == TouchPhase.Stationary)
                    {
                        
                        if (firstTouch == true && Time.time - touchTime > 0.6f)
                        {
                            firstTouch = false;
                            Swap(ref scale_begin, ref scale_end);
                            scale_cur = 0;
                        }
                    }

                    // 移动
                    else if(touch.phase == TouchPhase.Moved)
                    {
                        var delta = (Input.mousePosition.x - lastX);
                        if(Mathf.Abs(delta) < 0.09f)
                        {
                            if (firstTouch == true && Time.time - touchTime > 0.5f)
                            {
                                firstTouch = false;
                                Swap(ref scale_begin, ref scale_end);
                                scale_cur = 0;
                            }
                        }
                        else
                        {
                            firstTouch = false;
                            hitInfo.collider.gameObject.transform.Rotate(Vector3.up, -xSpeed * Time.deltaTime * delta);
                            lastX = Input.mousePosition.x;
                        }
                        
                        
                    }
                    // 单击
                    else // (touch.phase == TouchPhase.Ended)
                    {
                        if (firstTouch == true)
                        {
                            firstTouch = false;
                            hitInfo.collider.gameObject.SetActive(false);
                        }
                    }

                }
            }
            else
            {
                soldier.SetActive(true);
            }
            
        }
        else if(firstTouch)
        {
            firstTouch = false;
            soldier.SetActive(false);
        }
    }
}
