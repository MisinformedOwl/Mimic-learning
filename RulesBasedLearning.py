#                    COLLECTION OF RULES
class Rules():
    '''
    This class holds all of the rules. The reason for this not being in RulesBased is because of efficiency.
    Using this method i can pass more specific rule information to Rules so that the algorithm doesn't need to look at all possible values to get the correct one.
    '''
    rules = {} #Action : rules
    
    cordinates = [(x,y) for x in range(32) for y in range(32)]
    
    def __init__(self, actions):
        '''
        Creates rules for all possible combinations.
        
        Parameters:
            actions (list): of all possible actions recorded in dataCollection.py
        '''
        for a in actions:
            rulelist = []
            for c in self.cordinates:
                for a2 in actions:
                    rulelist.append(self.createRule(a, c, a2))
            self.rules.update({a : rulelist})
    
    def createRule(self, previous, cordinates, action):
        '''
            Creates a rule to be added to rulelist and then rules dictionary
            
            Parameters:
                previous (string): What was the previous action taken?
                cordiantes ((int,int)): The cordiantes of the action being taken.
                action (string): The action being taken now.
                
            Return:
                Rule: Created rule
        '''
        return Rule(previous,cordinates, action)

    def updateRuleList(self, rule, update): # [Could result in errors and not being able to update the rule.]
        '''
        This method updates the weightings of rules based on feedback from the model in CNNModel.py
        
        Parameters:
            rule (Rule): the rule being updated
            update (int): The reward/punishment for getting it right/wrong.
        '''
        rule.updateWeight(update)
    
    def getRule(self, previous, cords):
        '''
        Simply identifies the rule that fits the criteria and returns it.
        
        Parameters:
            previous (string): The previous action taken
            cords ([int, int]): The cordinates of the action being taken now.
        
        Return:
            Rule: The rule matching the description.
        '''
        rules = self.rules.get(previous)
        for r in rules:
            if r.cordinates == cords:
                return r

#%%                INDIVIDUAL RULES
class Rule():
    '''
    the individual rule class.
    This is a for the mostpart static class. Whose only job is to return the action it is storing.
    '''
    cordinates = ()
    previousAction = ""
    action = ""
    weight = 0
    
    def __init__(self, prev, cords, action):
        '''
        When first creating the rule the variables are set.
        
        Parameters:
            prev (string): The previous action made
            cords ([int,int]): the location of the action
            action (string): The action that will be taken should this rule be selected.
        '''
        self.cordinates = cords
        self.previousAction = prev
        self.action = action
    
    def getWeight(self):
        '''
        Returns the weight of the rule for comparison purposes.
        
        Returns:
            int: the weighting of the rule.
        '''
        return self.weight
    
    def updateWeight(self, update):
        '''
        Adds the update value to the weighting. (This can be negative)
        
        Parameters:
            update (int): The reward/punishment for the rule.
        '''
        self.weight += update

#%%                START
class RulesBased():
    '''
    Rules based is the master class of the other rule classes. This manages 
    which rule should be selected. as well as updating the weights of the rules.
    '''
    previousAction = ""
    locations = 0
    
    def __init__(self, actions, locations):
        '''
        when initialising the rules for the first time the Rules class is passed 
        a list of unique chacaters that appear during data collection.
        '''
        self.rules = Rules(actions)
        self.previousAction = actions[0]
        self.locations = locations
        
    def getRules(self, cordinates):
        '''
        Gets the rule based on location as well as the previous action taken.
        
        Return:
            Rule: The rule that matches the description.
        '''
        return self.rules.getRule(self.previousAction, cordinates)
    
    def decipherLocation(self, location):
        '''
        Changes location from being a [float,float] to a [int,int] of a small 
        size. This is to prevent the obserdly high amount of rules that would 
        be created when operating with precise numbers.
        
        Parameters:
            location ([float,float]): The location the CNNModel believes is correct.
        
        Return:
            Tuple: containing small integers pointing to a sector of the image.
        '''
        return (round(location[0]*self.locations), round(location[1]*self.locations))
    
    def weightTrain(self, location, actionUsed, label): #update weights based on correctness.
        '''
        This tests to see if the rule collected is correct by matching it to the expected button given in CNNModel.py's training data
        
        Parameters:
            location ([float],[float]): The location predicted by the CNNModel
            actionUsed (String): The action that was used previously. ; look into this.
            label (String): the expected action.
        '''
        location = self.decipherLocation(location)
        rule = self.getRules(location)
        if rule.actionUser == label:
            self.rules.updateRuleList(rule, 1)
        else:
            self.rules.updateRuleList(rule, -1)
    
        self.previousAction = actionUsed
    
    def checkRules(self,location, actionUsed): #used to see which rule should be used. So no updating
        '''
        This is the version used when not training. So the rules are not updated if incorrect.
        
        Parameters:
            location ([float,float]): the predicted location by the AI.
            actionUsed (String): The action that is used ; Again look into this, i dont see what the purpose of this is.
        '''
        location = self.decipherLocation(location)
        self.previousAction = actionUsed
        return self.getRules(location).weight
        
    def getPreviousAction(self):
        '''
        Returns the previous action that was used.
        
        Returns:
            String: the previous action used.
        '''
        return self.previousAction