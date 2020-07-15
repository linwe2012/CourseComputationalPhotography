using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
public class SwitchScene : MonoBehaviour
{
    public void GoToMain()
    {
        SceneManager.LoadScene("SampleScene");
    }

    public void GoToGame()
    {
        SceneManager.LoadScene("Game");

    }
}
