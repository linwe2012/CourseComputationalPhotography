using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Vuforia;
public class SwitchCar : MonoBehaviour, IVirtualButtonEventHandler
{
    public int target = 0;
    public GameObject control;
    public GameObject vbutton;
    // Start is called before the first frame update
    void Start()
    {
        vbutton.GetComponent<VirtualButtonBehaviour>().RegisterEventHandler(this);
    }

    public void OnButtonPressed(VirtualButtonBehaviour vb)
    {
        control.GetComponent<Swipe>().PickShowing(target);
    }
    public void OnButtonReleased(VirtualButtonBehaviour vb)
    {

    }

}
