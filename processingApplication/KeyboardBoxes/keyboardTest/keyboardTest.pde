class boxSeg {
  int x = 0;
  int y = 0;
  color col;
  String letter = "";
  
  boxSeg(int xi,int yi, color c){
  x=xi;
  y=yi;
  col = c;
  }
  
  void setLetter(String let){
  letter = let;
  }
  
  void resetLetter(){
  letter = "";
  }
  
  void draw(){
    fill(col);
    rect(x*(width/xmax), y*(height/ymax), width/xmax, height/ymax);
    fill(color(0,0,0));
    text(letter, x*(width/xmax)+((width/xmax)/2), y*(height/ymax)+((width/xmax)/2));
  }
  
  void updateColor(color c){
    col = c;
  }
  
  boolean hit(int mx, int my){
    markedx = x;
    markedy = y;
    if(
    mx > x*(width/xmax) &&
    mx < (x*(width/xmax) + (width/xmax)) &&
    my > y*(height/ymax) &&
    my < (y*(height/ymax) + (height/ymax))
    )
    {
    return true;
    } else {
    return false;
    }
  }
}

int xmax = 4;
int ymax = 4;
boxSeg[] boxes = new boxSeg[xmax*ymax];
String[] letters = {"s", "a", "d", "w"};
String selectedLetter = "";
boolean clicked = false;
boolean correct = false;
int xlet = 0;
int ylet = 0;
int count = 0;
int markedx = 0;
int markedy = 0;

void updateBoxes(){
  int count = 0;
  
  //Pick letter location
  xlet = (int)random(xmax);
  ylet = (int)random(ymax);
  String l = letters[(int)random(letters.length)];
  selectedLetter = l;
  
  for(int y = 0; y < ymax; y++){
    for(int x = 0; x < xmax; x++){
      color c = color(random(255),random(255), random(255));
      boxes[count].updateColor(c);
      if(xlet==x && ylet==y){
        boxes[count].setLetter(l);
      } else {
        boxes[count].setLetter("");
      }
      count++;
    }
  }
}

void keyPressed(){
  clicked = true;
  int index = ((xmax*ylet)+xlet);
  if (boxes[index].hit(mouseX,mouseY)){
    if (key == selectedLetter.charAt(0)){
    correct = true;
    updateBoxes();
    return;
    }
  }
  correct = false;
  updateBoxes();
  return;
}

void setup() {
  size(600,600);
  textSize((width/xmax)/2);
  color c;
  int count = 0;
  
  for(int x = 0; x < xmax; x++){
    for(int y = 0; y < ymax; y++){
      c = color(random(255), random(255), random(255));
      boxes[count] = new boxSeg(x,y,c);
      count++;
    }
  }
}

void draw(){
  if(clicked == false){
    for (boxSeg b : boxes){
      b.draw();
    }
  } else {
    if (correct) {
      background(0,255,0);
    } else {
      background(255,0,0);
      fill(255);
      rect(markedx*(width/xmax), markedy*(height/ymax), width/xmax, height/ymax);
    }
    count++;
    if (count > 60) {
      count = 0;
      clicked=false;
    }
  }
}
