import re
import os
import json
import random
import time
from string import ascii_lowercase
from word2number import w2n

#----------------------------------------------------------------------------------------------------
# DATA MANAGEMENT

# Generates new dictionaries for characters, symbols and words starting with each letter in new directories
def Generate():
    if not os.path.exists('dictionaries'):
        os.makedirs('dictionaries')

    if not os.path.exists('characters'):
        os.makedirs('characters')

    if not os.path.exists('samples.json'):
        with open('samples.json', 'w') as f:
            json.dump([], f)

    if not os.path.exists('dictionaries\sym.json'):
        with open('dictionaries\sym.json', 'w') as f:
            json.dump({}, f)

    for i in range(5):
        if not os.path.exists(f'characters\{i+1}.json'):
            with open(f'characters\{i+1}.json', 'w') as f:
                json.dump({'Greetings' : {}, 'Responses' : {}}, f)
    
    for c in ascii_lowercase:
        if not os.path.exists(f'dictionaries\{c}.json'):
            with open(f'dictionaries\{c}.json', 'w') as f:
                json.dump({},f)

# Cleans dictionaries of any words that have no definitions
def Clean():
    for c in ascii_lowercase:
        with open(f'dictionaries\{c}.json', 'r') as f:
            repo = json.load(f)
            for word in list(repo):
                if not repo[word]:
                    repo.pop(word, None)
        with open(f'dictionaries\{c}.json', 'w') as f:
            json.dump(repo, f)

# Records user inputs or responses as samples
def Sample(sentence):
    # Updates the samples to choose from
    with open('samples.json', 'r') as f:
        samples = json.load(f)
    if not sentence in samples:
        samples += [sentence]
    with open('samples.json', 'w') as f:
        json.dump(samples, f)

# Formats the sentence according to recorded samples if found, otherwise returns sentence untouched
def Format(sentence):
    with open('samples.json', 'r') as f:
        samples = json.load(f)
    for sample in samples:
        if sentence.lower() == sample.lower():
            sentence = sample
            return sentence

    return sentence

# Resorts the dictionaries by key
def Resort():
    for c in ascii_lowercase:
        with open(f'dictionaries\{c}.json', 'r') as f:
            repo = json.load(f)
        repo = dict(sorted(repo.items()))
        with open(f'dictionaries\{c}.json', 'w') as f:
            json.dump(repo, f)

# Expanded empty conditions
def Empty(variable):
    if not variable or variable is None or variable == [''] or variable == [['']]:
        return True
    else:
        return False

# Using logic to save time in adding information
def Logic(define, root, parameter, value):
    logic_one = ['Root', 'Derive', 'Meaning']
    logic_two = ['Meaning', 'Function', 'Synonyms', 'Antonyms', 'Root']

    if not Empty(value):
        return False
    
    if Empty(define) and Empty(root):
        if parameter == 'Meaning' or parameter == 'Root':
            return True
    if not Empty(define):
        if not parameter in logic_one:
            return True
    if not Empty(root):
        if not parameter in logic_two:
            return True

    return False
            
# Adds info to words in dictionaries
def Info_Add():
    str_values = ['Root', 'Meaning', 'Function', 'Sentence']
    list_values = ['Derive', 'Synonyms', 'Antonyms']
    
    for c in ascii_lowercase:
        with open(f'dictionaries\{c}.json', 'r') as f:
            repo = json.load(f)
        for word in repo:
            for meaning in repo[word]:
                for parameter in repo[word][meaning]:
                    define = repo[word][meaning]['Meaning']
                    root = repo[word][meaning]['Root']
                    value = repo[word][meaning][parameter]
                    if Logic(define, root, parameter, value):
                        if parameter in str_values:
                            print(f'{word.lower()}: {define}')
                            print(f'What is the {parameter.lower()} for {word.lower()}?')
                            value = input()
                        else:
                            print(f'{word.lower()}: {define}')
                            print(f'What are the {parameter.lower()} for {word.lower()}?')
                            value = input().split(' | ')
                            
                        repo[word][meaning][parameter] = value    
                        with open(f'dictionaries\{c}.json', 'w') as f:
                            json.dump(repo, f)
                            print('Updated!\n')

# Finds a word in the dictionary and reduces the amount of meanings down to some number
def Simplify(word, num_meanings):
    word_class = Word(word)
    index = word_class.Index()

    with open(f'dictionaries\{index}.json', 'r') as f:
        repo = json.load(f)
    try:
        for meaning in list(repo[word.upper()]):
            if int(meaning) > num_meanings:
                repo[word.upper()].pop(meaning, None)
    except KeyError:
        return f'Unable to find {word}!'

    print(repo)
    with open(f'dictionaries\{index}.json', 'w') as w:
        json.dump(repo, f)
    
            
#------------------------------------------------------------------------------------------------------
# CLASSES
            
# Handles the word
class Word():
    def __init__(self, word):
        self.word = word

    # Figures out if the word is actually math related, then handles that separately
    def Math(self):
        symbols = ['+', '-', '*', '/', '%', '^', '//']
        math = {'1': {
            'Root' : None,
            'Derive' : None,
            'Meaning' : 'Math',
            'Function' : None,
            'Sentence' : None,
            'Synonyms' : None,
            'Antonyms' : None
            }}
        
        try:
            number = w2n.word_to_num(self.word)
            if type(number) == int or type(number) == float:
                return math
        except ValueError:           
            if self.word in symbols or self.word.isdigit():
                return math
            else:
                return False
            
    # Finds the first letter of the word, return 'sym' if it's a symbol    
    def Index(self):
        index = self.word[0].lower()
        if not index.isalpha():
            index = 'sym'
        return index

    # Looks for the word in another dictionary, inserting and returning 1 that if found
    def Harvest(self):
        ins = {}
        index = self.word[0].upper()
    
        if index.isalpha():
            with open(f'data\D{index}.json', 'r') as f:
                repo = json.load(f)
                
            word_dict = repo[self.word.upper()]
            meanings = word_dict['MEANINGS']
            if not meanings:
                raise KeyError
                
            i = 0
            for key in meanings.keys():
                ins[str(i+1)] = {
                    'Root' : None,
                    'Derive' : None,
                    'Meaning' : meanings[key][1],
                    'Function' : meanings[key][0],
                    'Sentence' : None,
                    'Synonyms' : word_dict['SYNONYMS'],
                    'Antonyms' : word_dict['ANTONYMS']
                    }
                i += 1

        return ins

    # Inserts the word's meaning into the dictionary
    def Insert(self):
        meaning, function, sentence, synonyms, antonyms, derives = [], [], [], [], [], []
        root = None
        ins = {}
        index = self.Index()
        
        print(f'Does the word "{self.word}" derive from a root? (Yes/No)')
        if input().lower() == 'yes':
            print(f'What is the root of the word?')
            root = input()
            print(f'What kind of derivation is this?')
            derives = input().split(' | ')
            print(f'Can you use the word in a sentence?')
            sentence.append(input())

            ins['1'] = {
                'Root' : root,
                'Derive' : derives,
                'Meaning' : meaning,
                'Function' : function,
                'Sentence' : sentence,
                'Synonyms' : synonyms,
                'Antonyms' : antonyms
                }

        else:
            try:
                ins = self.Harvest()
                for meaning in ins:
                    print(f'{meaning}: {ins[meaning]["Meaning"]}')
                print(f'Would you like to use a harvested definition for {self.word}?')
                if input().lower() == 'yes':
                    if len(ins) > 1:
                        print(f'{self.word} has {len(ins)} meanings. Which ones do you want? eg. 1 2 3')
                        take = input().split()
                        for meaning in ins:
                            if int(meaning) in take:
                                ins.pop(meaning, None)
                else:
                    raise KeyError
                        
            except KeyError:
                meaning = []
                while True:
                    print(f'What is the meaning of the word \"{self.word}\"?')
                    meaning.append(input())
                    print('What function does this word have in a sentence?')
                    function.append(input())
                    print('Can you use the word in a sentence?')
                    sentence.append(input())
                    print('Any synonyms for this word?')
                    synonyms.append(input().split(' | '))
                    print('Any antonyms?')
                    antonyms.append(input().split(' | '))                
                
                    for i in range(len(meaning)):
                        ins[str(i+1)] = {
                            'Root' : root,
                            'Derive' : derives,
                            'Meaning' : meaning[i],
                            'Function' : function[i],
                            'Sentence' : sentence[i],
                            'Synonyms' : synonyms[i],
                            'Antonyms' : antonyms[i]
                            }

                    print('Are there any other meanings of this word? (Yes/No)')
                    if input().lower() != 'yes':
                        break

        with open(f'dictionaries\{index}.json', 'r') as f:
            repo = json.load(f)
        repo[self.word.upper()] = ins
        with open(f'dictionaries\{index}.json', 'w') as f:
            json.dump(repo, f)

    # Finds the meaning of the word            
    def Find(self):
        math = self.Math()
        if self.Math():
            return math
        
        index = self.Index()
        with open(f'dictionaries\{index}.json', 'r') as f:
            repo = json.load(f)
        try:
            return repo[self.word.upper()]
        except KeyError:
            f.close()
            self.Insert()
            with open(f'dictionaries\{index}.json', 'r') as f:
                repo = json.load(f)
                return repo[self.word.upper()]
                
# Handles the sentence
class Sentence():
    def __init__(self, sentence):
        self.sentence = sentence
        
    # Finds words in a sentence, also splitting up contractions like 'm, 's and n't
    def Words(self):
        s = self.sentence
        
        for i in range(len(s)):
            if s[i] == "\'":
                if s[i-1] == 'n' and s[i+1] == 't':
                    # Can't is the only one that splits into "ca" and "n't"
                    if s[i-3:i].lower() == 'can':
                        s = s[:i] + "~n\'t" + s[i+2:]
                        i += 3
                    else:
                        s = s[:i-1] + "~n\'t" + s[i+2:]
                        i += 3
                else:
                    s = s[:i] + "~\'" + s[i+1:]
                    i += 2
            else:
                i += 1
            
        words = re.findall(r"[\w']+|[\s]+|[-]+|[.,!?;]", s)
        return words

    # Finds the meaning of the words in the sentence
    def Find(self):
        sentence = {}
        words = self.Words()
        
        for word in words:
            word_dict = Word(word)
            sentence[word] = word_dict.Find()

        return sentence

# Handles the character, or the personalized weights of each person in the database.
class Character():
    def __init__(self, character):
        self.character = character

    # Determines if the character will learn this session
    def Learning(self):
        global pgate
        
        print(f'Do you want {self.character} to learn during this session? (Yes/No)')
        answer = input()
        if answer.lower() == 'yes':
            pgate = 1
        else:
            pgate = 0

    # Determines what phrase a character would prefer from a list of phrases
    def Weighted_Selection(self, pref):
        global pgate
        keys, values, trials = [], [], []
        
        for key in pref:
            keys += [key]
            values += [pref[key][0]]
            trials += [pref[key][1]]

        # Determines if we try a different phrase that hasn't been recorded.
        # This gives us more samples to perform the HMM on later.
        avg_value = sum(values) / len(values)
        avg_trial = sum(trials) / (1 + sum(trials))
        avg_weight = avg_value * avg_trial
        gate = random.choices([1, 0], [avg_weight, 1-avg_weight], k=1)[0]
        if gate == 0 and pgate == 1:
            values = [0.0]
            
        phrase = random.choices(keys, values, k=1)[0]
        phrase = Format(phrase)      
        return phrase        

    # Determines if the character will greet the user, and how
    def Greeting(self):
        score, trials = 0, 0
        full = []
        
        with open(f'characters\{self.character}.json', 'r') as f:
            pref = json.load(f)
        try:
            greetings = pref['Greetings']
            greeting = self.Weighted_Selection(greetings)
            return greeting
        except (ValueError, ZeroDivisionError):
            try:
                responses = pref['Responses']
                for response in responses:
                    full.append(response)
                response = random.choice(full)
                greeting = Format(response)
                return greeting
            except KeyError:
                return
                
    # Handles how the character will use synonyms in their response
    def Change(self, response, response_meaning):
        change = ''
        sentence = Sentence(response)
        words = sentence.Words()
        
        for word in words:
            try:
                synonyms = response_meaning[word]['Synonyms']
                words = [word] + synonyms
                change += random.choice(words)
            except KeyError:
                change += word

        return change

    # Returns best response to input
    def First_State(self, user, pref):      
        try:
            responses = pref['Responses'][user.lower()]
            return self.Weighted_Selection(responses)
        
        except (ValueError, ZeroDivisionError):
            return self.Second_State(user, pref)
        
        except KeyError:
            return self.Third_State(user, pref)
        
    # Returns different response to input
    def Second_State(self, user, pref):
        psb = []
        responses = pref['Responses']
        
        for string in responses:
            if string != s.lower():
                psb += [string]
        if psb:
            response = psb[random.randint(0, len(psb)-1)]
        else:
            response = user
            
        return response
            
    # Returns any input to input
    def Third_State(self, user, pref):
        keys, values = [], []
        score = {}

        try:
            responses = pref['Responses']
            for string in responses:
                score[string] = 0
                words = re.findall(r"[\w']+|[.,!?;]", string)
                for word in words:
                    if word in user.lower():
                        score[string] += 1

            closest = max(score, key=score.get)
            response = responses[closest]
            for key in response:
                keys += [key]
                values += [response[key][0]]
            if len(values) == 1 and values[0] == 0:
                values[0] = 1.0
            response = random.choices(keys, values, k=1)[0]
            return response
        except (ValueError, KeyError):
            return response
        
    # Handles how the character should respond to the user's input.   
    def Response(self, user):
        user_meaning, response_meaning = {}, {}
        user_sentences = re.findall(r"[\w'\s]+|[.,!?;\s]+", user)

        # Finds the meaning of the user input
        for user_sentence in user_sentences:
            user_sentence_dict = Sentence(user_sentence)
            user_meaning.update(user_sentence_dict.Find())

        with open(f'characters\{self.character}.json', 'r') as f:
            pref = json.load(f)
        response = self.First_State(user, pref)
        response_sentences = re.findall(r"[\w'\s]+|[.,!?;\s]+", response)
        for response_sentence in response_sentences:
            response_sentence_dict = Sentence(response_sentence)
            response_meaning.update(response_sentence_dict.Find())

        response = self.Change(response, response_meaning)
        response = Format(response)    
        return response

    # Handles scoring of response to user's input.
    # Add more scoring details to check for other parameters and score more accurately. 
    def Score(self, user, response, score):
        score = int(score)/10

        with open(f'characters\{self.character}.json', 'r') as f:
            pref = json.load(f)
            
        if user:
            try:
                prev_score = pref['Responses'][user.upper()][response]
                prev_score[1] += 1
                prev_score[0] = round((score + prev_score[0]) / prev_score[1], 3)
            except KeyError:
                pref['Responses'][user.upper()] = {response : [score, 1]}
        else:
            try:
                prev_score = pref['Greetings'][response.upper()]
                prev_score[1] += 1
                prev_score[0] = round((score + prev_score[0]) / prev_score[1], 3)
            except KeyError:
                pref['Greetings'][response.upper()] = [score, 1]
                
        with open(f'characters\{self.character}.json', 'w') as f:
            json.dump(pref, f)

    # Adds some sample responses to character.
    def Driver(self):
        full = {}
        
        with open(f'characters\{self.character}_driver.txt', 'r') as f:
            pref = f.read()
            pref = pref.strip()
            responses = pref.split('||')
            for key_value in responses:
                key_value = key_value.split(' : ')
                key = key_value[0].strip()
                values = key_value[1].split(' | ')
                full[key] = {}
                for value in values:
                    value = value.strip()
                    full[key].update({value : [1.0, 1]})

        with open(f'characters\{self.character}.json', 'r') as f:
            pref = json.load(f)
        pref['Responses'].update(full)
        with open(f'characters\{self.character}.json', 'w') as f:
            json.dump(pref, f)

#----------------------------------------------------------------------------------------------------
# MAIN
        
# Handles the allocation of user inputs and responses.
def main():
    global pgate
    Generate()
    Clean()
    Resort()
    
    try:
        print('Who would you like to talk to? (1-5)')
        char = input()
        if not char.isdigit and 1<=int(char)<=5:
            print("Not a character!")
        else:
            character = Character(char)
            character.Learning()
            greeting = character.Greeting()
            if greeting:
                print(f'{char}: ' + greeting)
                if pgate == 1:
                    # Scoring of Greeting
                    print('Scoring? (0-10): ')
                    character.Score(None, greeting, input())

                    # Storing Greeting
                    Sample(greeting)

            while True:
                # User input
                print('User: ')
                user = input()

                # Response to User Input
                response = character.Response(user)
                print(f'{char}: ' + response)

                if pgate == 1:
                    # Scoring of Response
                    print('Scoring? (0-10): ')
                    character.Score(user, response, input())

                    # Storing User Input and Response
                    Sample(user)
                    Sample(response)
                
    except KeyboardInterrupt:
        print('Closing...')

if __name__ == "__main__":
    main()
        
    
