using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Swipe : MonoBehaviour
{

    float energy = 0;
    GameObject car;
    Vector3 forward = Vector3.right;
    float speed = 0;
    int showing = 0;
    string[] targets =
    {
        "Raider",
        "Cool",
        "Truck"
    };

    List<GameObject> tarobj = new List<GameObject>();

    public void PickShowing(int n)
    {
        showing = n;
        car = tarobj[n];
        speed = energy = 0f;
        for (int i =0; i < targets.Length; ++i)
        {
            if(i == n)
            {
                car.SetActive(true);
            }
            else
            {
                tarobj[i].SetActive(false);
                // GameObject.Find("ImageTarget/" + targets[i]).SetActive(false);
            }
        }
    }
    
    // Start is called before the first frame update
    void Start()
    {
        foreach (var target in targets)
        {
            tarobj.Add(GameObject.Find("ImageTarget/" + target));
        }
        // car = GameObject.Find("ImageTarget/Raider");
        PickShowing(1);
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetMouseButton(0))
        {
            Touch touch = Input.GetTouch(0);
            if(touch.phase == TouchPhase.Began)
            {
                //firstMouse = true;
            }

            else if(touch.phase == TouchPhase.Moved)
            {
                float dx = Vector2.Dot(touch.deltaPosition, new Vector2(forward.x, forward.z));
                energy += dx * 50;
                car.transform.Translate(0, dx / 1250.0f, 0);
            }
            //Vector3 delta = 
        }
        
        else if(Mathf.Abs(energy) > 0.00101f)
        {
            speed += energy * 0.3f;
            energy *= 0.7f;

            car.transform.Translate(0, -speed * Time.deltaTime / 10000.0f, 0);

            if(Mathf.Abs(speed) < 0.025f)
            {
                speed = 0f;
            }
            else
            {
                speed = Mathf.Sign(speed) * Mathf.Abs((Mathf.Abs(speed) - 0.05f) * 0.95f);
            }
        }
        
    }
}
