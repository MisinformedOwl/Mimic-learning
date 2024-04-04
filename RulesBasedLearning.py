#                    COLLECTION OF RULES
class Rules():
    
    rules = {} #Action : rules
    
    cordinates = [(x,y) for x in range(32) for y in range(32)]
    
    def __init__(self, actions):
        
        for a in actions:
            rulelist = []
            for c in self.cordinates:
                for a2 in actions:
                    rulelist.append(self.createRule(a, c, a2))
            self.rules.update({a : rulelist})
    
    def createRule(self, previous, cordinates, action):
        return Rule(previous,cordinates, action)

    def updateRuleList(self, rule, update): # [Could result in errors and not being able to update the rule.]
        rule.updateWeight(update)
    
    def getRule(self, previous, cords): 
        rules = self.rules.get(previous)
        for r in rules:
            if r.cordinates == cords:
                return r

#%%                INDIVIDUAL RULES
class Rule():
    cordinates = ()
    previousAction = ""
    action = ""
    weight = 0
    
    def __init__(self, prev, cords, action):
        self.cordinates = cords
        self.previousAction = prev
        self.action = action
    
    def getWeight(self):
        return self.weight
    
    def updateWeight(self, update):
        self.weight += update

#%%                START
class RulesBased():
    
    previousAction = ""
    locations = 0
    
    def __init__(self, actions, locations):
        self.rules = Rules(actions)
        self.previousAction = actions[0]
        self.locations = locations
        
    def getRules(self, cordinates):
        return self.rules.getRule(self.previousAction, cordinates)
    
    def decipherLocation(self, location):
        return (round(location[0]*self.locations), round(location[1]*self.locations))
    
    def weightTrain(self, location, actionUsed, label): #update weights based on correctness.
        location = self.decipherLocation(location)
        rule = self.getRules(location)
        if rule.actionUser == label:
            self.rules.updateRuleList(rule, 1)
        else:
            self.rules.updateRuleList(rule, -1)
    
        self.previousAction = actionUsed
    
    def checkRules(self,location, actionUsed): #used to see which rule should be used. So no updating
        location = self.decipherLocation(location)
        return self.getRules(location).weight
        self.previousAction = actionUsed
        
    def getPreviousAction(self):
        return self.previousAction