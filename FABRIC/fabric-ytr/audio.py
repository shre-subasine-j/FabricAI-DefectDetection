import win32com.client
speaker = win32com.client.Dispatch("SAPI.SpVoice")
speaker.Speak("Jumpman Jumpman Jumpman Them boys up to something!")