using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Vuforia;
using UnityEngine.SceneManagement;

public class ClickSwitch : MonoBehaviour, IVirtualButtonEventHandler
{
    public string target = "Game";
    public GameObject vbutton;
    public void OnButtonPressed(VirtualButtonBehaviour vb)
    {
        SceneManager.LoadScene(target);
    }
    public void OnButtonReleased(VirtualButtonBehaviour vb)
    {

    }

    void Start()
    {
        vbutton.GetComponent<VirtualButtonBehaviour>().RegisterEventHandler(this);
    }

}
