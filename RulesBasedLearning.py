from torch import round as rounding

#                    COLLECTION OF RULES
class Rules():
    '''
    This class holds all of the rules. The reason for this not being in RulesBased is because of efficiency.
    Using this method i can pass more specific rule information to Rules so that the algorithm doesn't need to look at all possible values to get the correct one.
    '''
    rules = {} #Action : rules
    
    
    def __init__(self, actions, cords):
        '''
        Creates rules for all possible combinations.
        
        Parameters:
            actions (list): of all possible actions recorded in dataCollection.py
            cords (int): Number of segments of the image to be used. Evently distributed.
        '''
        
        self.cordinates = [(x,y) for x in range(cords) for y in range(cords)]
        for a in actions:
            rulelist = {}
            for c in self.cordinates:
                rulelist.update({str(c) : [self.createRule(a, c, a2) for a2 in actions]})
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
            cords ((int, int)): The cordinates of the action being taken now.
        
        Return:
            Rule: The rule matching the description.
        '''
        print(cords)
        maxWeight = -99
        maxindex = 0
        rules = self.rules.get(previous).get(str(cords))
        for r in range(len(rules)):
            if rules[r].weight > maxWeight:
                maxWeight = rules[r].weight
                maxindex = r
        return rules[maxindex]
                

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
            cords ((int,int)): the location of the action
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
    success=0
    fail = 0
    
    def __init__(self, actions, locations):
        '''
        when initialising the rules for the first time the Rules class is passed 
        a list of unique chacaters that appear during data collection.
        
        Parameters:
            actions (Set): A set of unique actions which the user has performed.
            locations int
        '''
        self.rules = Rules(actions, locations+1)
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
        return (int(rounding(location[0]*self.locations, decimals=0).clamp(min=0, max =self.locations).item()), int(rounding(location[1]*self.locations, decimals=0).clamp(min=0, max =self.locations).item()))
    
    def weightTrain(self, location, label): #update weights based on correctness.
        '''
        This tests to see if the rule collected is correct by matching it to the expected button given in CNNModel.py's training data
        
        Parameters:
            location ([[float, float],...]): The location predicted by the CNNModel
            label ([String]): the expected action.
        '''
        for l, lab in zip(location,label):
            loc = self.decipherLocation(l)
            rule = self.getRules(loc)
            if rule.action == lab:
                self.rules.updateRuleList(rule, 1)
                self.success +=1
            else:
                self.rules.updateRuleList(rule, -1)
                self.fail +=1
            print(rule.action, lab, rule.action == lab)
        
            self.previousAction = rule.action
        return rule.action
    
    def checkRules(self,location): #used to see which rule should be used. So no updating
        '''
        This is the version used when not training. So the rules are not updated if incorrect.
        
        Parameters:
            location ([[float,float],...]): the predicted location by the AI.
            actionUsed (String): The action that is used ; Again look into this, i dont see what the purpose of this is.
        '''
        for l in location:
            location = self.decipherLocation(l)
            
            rule = self.getRules(location)
            
            self.previousAction = rule.action
        return rule.action
        
    def getPreviousAction(self):
        '''
        Returns the previous action that was used.
        
        Returns:
            String: the previous action used.
        '''
        return self.previousAction
    
    def accuracy(self):
        '''
        Returns the overall accuracy of the model so far.
        
        Returns:
            float: The accuracy xx.x%
        '''
        return (self.success/self.fail)*100