import tello
import time
import winsound

if __name__ == '__main__':
    print("starting...")
    my_drone = tello.Tello()
    print(my_drone.get_battery())
    print(my_drone.get_temp())
    my_drone.streamon()
    print("about to take off...")
    my_drone.takeoff()
    my_drone.up(50)
    time.sleep(3)
    #my_drone.forward(80)
    try:
        my_drone.processMoveCmd()
    except KeyboardInterrupt:
        print("about to land...")
        my_drone.land()
        my_drone.streamoff()
        print(my_drone.get_battery())
        print(my_drone.get_temp())
        quit()