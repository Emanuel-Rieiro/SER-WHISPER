from src.rangos import ekman, occ, russell_mehrabian

def emocion_ekman(valence : float, arousal : float, dominance : float) -> str:

    best_emotion = None
    min_distance = float('inf')

    # Calculate the Euclidean distance from the VAD values to each emotion's threshold
    for emotion, (v_thresh, a_thresh, d_thresh) in ekman.items():
        distance = (((valence / 100) - v_thresh) ** 2 + ((arousal / 100) - a_thresh) ** 2 + ((dominance / 100) - d_thresh) ** 2) ** 0.5 
        
        # Update the best emotion if the current emotion is closer
        if distance < min_distance:
            best_emotion = emotion
            min_distance = distance 
    
    return best_emotion

#"""
# Wilton Beltré
#"""
#class VAD:
#    def __init__(self, minmax=[1, 5], increase_interv=0., mapping="Russell_Mehrabian"): # mapping="Ekman"
#        """
#        Simple mapper class from vad model to categorical.
#        based on paper: Evidence for a Three-Factor Theory of Emotions.
#        mapping:
#                - Russell_Mehrabian categorical values (151 emotions)
#                - OCC: Translation of the ALMA model,  A Layered Model of Affect part of the Virtual Human Project
#                - Ekman: 6 basic emotions
#
#        """
#
#        current_path = Path(__file__).parent.__str__()
#
#        data = np.genfromtxt(f'{current_path}/categorial_vad.csv', delimiter=',', usecols=(2, 3, 4, 5, 6, 7))
#        if mapping == "Russell_Mehrabian":
#            self.vad = np.genfromtxt(f'{current_path}/categorial_vad.csv', delimiter=',', usecols=(2, 4, 6))
#            self.terms = np.genfromtxt(f'{current_path}/categorial_vad.csv', delimiter=',', usecols=(0), dtype=None, encoding='utf-8-sig')
#        elif mapping == "OCC":
#            self.vad = np.genfromtxt(f'{current_path}/categorical_occ.csv', delimiter=',', usecols=(1, 2, 3))
#            self.terms = np.genfromtxt(f'{current_path}/categorical_occ.csv', delimiter=',', usecols=(0), dtype=None, encoding='utf-8-sig')
#        else:
#            self.vad = np.genfromtxt(f'{current_path}/categorical_ekman.csv', delimiter=',', usecols=(1, 2, 3))
#            self.terms = np.genfromtxt(f'{current_path}/categorical_ekman.csv', delimiter=',', usecols=(0), dtype=None, encoding='utf-8-sig')
#
#        self.valence_mean = data[:, 0]
#        self.valence_std = data[:, 1]
#        self.arousal_mean = data[:, 2]
#        self.arousal_std = data[:, 3]
#        self.dominance_mean = data[:, 4]
#        self.dominance_std = data[:, 5]
#
#        self.minmax = minmax
#        self.increase_interv = increase_interv
#        self.to_plot = None
#
#    def plot(self, title, template='plotly_dark', w=800, h=600):
#        df = self.to_plot
#        fig = px.scatter_3d(df, x='Valence', y='Arousal', z='Dominance',
#                            color='Terms', symbol='Terms', text='Info', size='ivClosest',
#                            template=template, title=title,
#                            height=h, width=w)
#        fig.show()
#
#    def __normalize(self, value: int) -> set:
#        """
#        normalize interval [a, b] to [-1, 1] to easy map to vad model.
#        also add min interval, max interval proportional
#        :param value:
#        :param minmax:
#        :return: min: interval, norm: middle interval, max: inverval
#        """
#        norm = 2 * ((value - self.minmax[0]) / (self.minmax[1] - self.minmax[0])) - 1
#        max = norm + (1./(self.minmax[1]-1)) + (norm * self.increase_interv)
#        min = norm - (1./(self.minmax[1]-1)) - (norm * self.increase_interv)
#        return min, norm, max
#
#    def __intervals(self, v: int, a: int, d: int) -> []:
#        """
#        prepare V,A,D intervals and normalized value
#        :param v:
#        :param a:
#        :param d:
#        :return: v_inter, a_inter, d_inter
#        """
#        v_inter: set = self.__normalize(v)
#        a_inter: set = self.__normalize(a)
#        d_inter: set = self.__normalize(d)
#        return [v_inter, a_inter, d_inter]
#
#    def vad2categorical(self, v: int, a: int, d: int, k: int = 10, use_plot: bool = False):
#        """
#            k -> top k elements
#        :param use_plot:
#        :param v:
#        :param a:
#        :param d:
#        :param k:
#        :return:
#        """
#        DISTANCE_COLUMN = 4
#        interval = self.__intervals(v, a, d)
#        vad_orig = np.array([interval[0][1], interval[1][1], interval[2][1]])
#        z = np.linalg.norm(self.vad - vad_orig, axis=1)
#        t = np.hstack((self.terms[:, np.newaxis], self.vad), dtype=object)
#        z = np.hstack((t, z[:, np.newaxis]))
#        z = z[z[:, DISTANCE_COLUMN].argsort()]
#
#        ranking = []
#        for i, m in enumerate(z):
#            if i == k and not use_plot: break
#            idx = np.where(self.terms == m[0])[0][0] + 1 # 0 in CTC is 'the blank'
#            ranking.append({'term': m[0], 'index': idx, 'closest': m[4], 'v': m[1], 'a': m[2], 'd': m[3]})
#
#        ranking = ranking[:k]
#
#        if use_plot:
#            self.to_plot = pd.DataFrame({
#                'Terms': [v['term'] for v in ranking],
#                'Valence': [v['v'] for v in ranking],
#                'Arousal': [a['a'] for a in ranking],
#                'Dominance': [d['d'] for d in ranking],
#                'Closest': [np.round(c['closest'], 4) for c in ranking],
#                'Info': [f"{c['term']} - {np.round(c['closest'], 4)}" for c in ranking],
#            })
#            self.to_plot['ivClosest'] = self.to_plot[['Closest']][::-1].reset_index(drop=True)
#
#        return ranking[:k], {'using_dominance': True}
#
#
#if __name__ == "__main__":
#    v, a, d = 1, 4, 1
#    # vad = VAD(mapping="Ekman")
#    vad = VAD(minmax=[-100, 100], mapping="OCC")
#    # r = vad.vad2categorical(5.99, 42.59, 22.3, k=1, use_plot=False)
#    r = vad.vad2categorical(5.99, 42.59, 22.3, k=3, use_plot=False)
#    print(r)
#    # vad.plot(title=f"Mapping {v},{a},{d} to (Russell_Mehrabian 151) categorical", w=1300, h=900)