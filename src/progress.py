import time

class Bar():
    def __init__(self, size:int = 50):
        self.size = size
        self.progress(0.)

    def progress(self, percentage:float):
        percentage = 0. if percentage < 0 else percentage
        percentage = 1. if percentage > 1 else percentage

        print("\r", end="")

        bar_str = ""
        for i in range(1, int(percentage*self.size)+1):
            bar_str += "▮"
        for i in range(int(percentage*self.size)+1, self.size+1):
            bar_str += " "
        
        bar_str += "| {0:.2f}%".format(percentage * 100)
        print(bar_str, end="")

        #if percentage >= 1:
        #    print()

    def delete(self):
        print("\r", end="")
        blankStr = ""
        for i in range(self.size+10):
            blankStr += " "
        print(blankStr, end="")
        print("\r", end="")

if __name__ == "__main__":
    bar = Bar()
    for i in range(0, 200+1):
        bar.progress(i / 200)
        time.sleep(0.0001)
    print("finish")