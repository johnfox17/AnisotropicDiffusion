
import paperDiscretization 
import PDDODiscretization



def main():
    method1 = paperDiscretization.paperDiscretization()
    method1.solve()

    method2 = PDDODiscretization.PDDODiscretization()
    method2.solve()


if __name__ == "__main__":
    main()
