from src.rangos import ekman, occ, russell_mehrabian

def obtener_emocion(valence : float, arousal : float, dominance : float, mapping : str) -> str:
    """
        Inputs:
            -valence: valor de valencia
            -arousal: valor de arousal
            -dominance: valor de dominance
            -mapping: tipo de mapeo a ser utilizado (Russell_Mehrabian, OCC, Ekman)
        Output:
            String de la mejor emoción
    """
    best_emotion = None
    min_distance = float('inf')

    if mapping == 'Russell_Mehrabian': emo_dict = russell_mehrabian
    elif mapping == 'OCC': emo_dict = occ
    elif mapping == 'Ekman': emo_dict = ekman
    else: raise Exception("Mapeado no reconocido")

    # Calculate the Euclidean distance from the VAD values to each emotion's threshold
    for emotion, (v_thresh, a_thresh, d_thresh) in emo_dict.items():
        distance = (((valence / 100) - v_thresh) ** 2 + ((arousal / 100) - a_thresh) ** 2 + ((dominance / 100) - d_thresh) ** 2) ** 0.5 
        
        # Update the best emotion if the current emotion is closer
        if distance < min_distance:
            best_emotion = emotion
            min_distance = distance 
    
    return best_emotion