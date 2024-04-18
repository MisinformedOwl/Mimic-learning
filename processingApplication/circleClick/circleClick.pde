void setup(){
  size(600,600);
  background(0);
  fill(255);
}

int boxX = 0;
int boxY = 0;
int circleX = 0;
int circleY = 0;
int shapeSize = 50;

void mouseClicked(){
  boxX = (int) random(width-shapeSize/2);
  boxY = (int) random(height-shapeSize/2);
  
  circleX = (int) random(width-shapeSize/2);
  circleY = (int) random(height-shapeSize/2);
}

void draw(){
  background(0);
  rect(boxX,boxY,shapeSize,shapeSize);
  ellipse(circleX,circleY,shapeSize,shapeSize);
}
