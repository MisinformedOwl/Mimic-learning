void setup(){
  size(600,600);
  background(0);
  fill(255);
}

int boxX = 200;
int boxY = 300;
int circleX = 400;
int circleY = 300;
int shapeSize = 50;
int counter = 0;
boolean correct = false;
boolean clicked = false;

void mouseClicked(){
  if((sq(mouseX - circleX) + sq(mouseY - circleY))< shapeSize*10){
    correct = true;
  } else {
    correct = false;
  }
  clicked = true;
  boxX = (int) random(width-shapeSize/2);
  boxY = (int) random(height-shapeSize/2);
  
  circleX = (int) random(width-shapeSize/2);
  circleY = (int) random(height-shapeSize/2);
}

void draw(){
  if(clicked){
    if(correct){
      background(0,255,0);
    } else {
      background(255,0,0);
    }
    counter++;
    if (counter > 60) {
      clicked = false;
      counter = 0;
    }
  } else {
    background(0);
    rect(boxX,boxY,shapeSize,shapeSize);
    ellipse(circleX,circleY,shapeSize,shapeSize);
  }
}
