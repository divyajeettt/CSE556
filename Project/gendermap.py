import json


class GenderMap:
    def __init__(self, path: str):
        with open(path) as file:
            json_data = json.loads(file.read())
        self.mapping = dict()
        for word in json_data:
            try:
                if word["gender"] == "m":
                    m_word = word["word"]
                    f_word = word["gender_map"]["f"][0]["word"]
                    self.mapping[m_word] = f_word
                    self.mapping[f_word] = m_word
                elif word["gender"] == "f":
                    f_word = word["word"]
                    m_word = word["gender_map"]["m"][0]["word"]
                    self.mapping[m_word] = f_word
                    self.mapping[f_word] = m_word
            except KeyError:
                pass

    def flip(self, sentence: str) -> str:
        words = sentence.split(" ")
        flipped_words = [self.mapping.get(word, word) for word in words]
        return " ".join(flipped_words)


if __name__ == "__main__":
    gm = GenderMap("gendered_words/gendered_words.json")
    print(gm.map["her"])