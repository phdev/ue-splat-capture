(event EventBeginPlay
  (Variables|Default|SetLastLocation (Transformation|GetActorLocation :self self)))

(event EventTick (DeltaSeconds)
  (bind cur (Transformation|GetActorLocation :self self))
  (Transformation|SetWorldRotation
    :self (Variables|CameraActor|GetCameraComponent)
    :NewRotation (Math|Rotator|MakeRotFromX :X (- (Variables|Default|GetLastLocation) cur)))
  (Variables|Default|SetLastLocation cur))
