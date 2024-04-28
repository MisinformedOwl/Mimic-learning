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
}

int xmax = 4;
int ymax = 4;
boxSeg[] boxes = new boxSeg[xmax*ymax];
String[] letters = {"s", "a", "d", "w"};

void updateBoxes(){
  int count = 0;
  
  //Pick letter location
  int xlet = (int)random(xmax);
  int ylet = (int)random(ymax);
  String l = letters[(int)random(letters.length)];
  
  for(int x = 0; x < xmax; x++){
    for(int y = 0; y < ymax; y++){
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

void mouseClicked(){
  updateBoxes();
}

void keyPressed(){
  updateBoxes();
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
  for (boxSeg b : boxes){
    b.draw();
  }
}
