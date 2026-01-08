from ai2thor.controller import Controller

controller = Controller()
event = controller.step("MoveAhead")

input("Unity is running. Press Enter to close...\n")

controller.stop()
