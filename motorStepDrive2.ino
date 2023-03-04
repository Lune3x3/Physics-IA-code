bool run = false;

void setup() {
  // put your setup code here, to run once:
  pinMode(2, OUTPUT);
  pinMode(3, OUTPUT);
  digitalWrite(2, HIGH);
}

void step() {
  digitalWrite(3, HIGH);
  digitalWrite(3, LOW);
  delay(250);
}

void loop() {
  // put your main code here, to run repeatedly:
  if(run == false){
    for (long i = 0; i < 800; i++) {
      step();
    }
    run = true;
  }
}
