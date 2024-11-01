

PROBLEMS = {
    "G1" : {
        "Equation" : lambda x: 5*x[0] + 5*x[1] + 5*x[2] + 5*x[3] - 5*sum(x[0:4]) - sum(x[4:]),
        "Constraints" : [
            2*x1 + 2*x2 + x10 + x11 <= 10,
            -8*x1 + x10 <= 0,
            -2*x4 - x5 + 10 <= 0,
            2*x1 + 2*x3 + x10 + x12 <= 10,
            -8*x2 + x11 <= 0,
            -2*x6 - x7 + x11 <= 0,
            2*x2 + 2*x3 + x11 + x12 <= 10,
            -8*x3 + x12 <= 0,
            -2*x8 - x9 + x12 <= 0
        ],
        "Optimal" : {
            "Solution" : [
                1,1,1,1,1,1,1,1,1,3,3,3,1
            ],
            "Evaluation" : 15
        }
    },

    "G4" : {
        "Equation" : lambda x: 5.3578547*x[2]**2 + 0.8356891*x[0]*x[4] - 40792.141,
        "Constraints" : [
            0 <= 85.334407 + 0.0056858*x2*x5 + 0.00026*x1*x4 - 0.0022053*x3*x5 <= 92,

        ],
        "Optimal" : {
            "Solution" : [
                78.0, 33.0, 29.995, 45.0, 36.776
            ],
            "Evaluation" : -30665.5
        }
    },

    "G5" : {
        "Equation" : lambda x: 3*x[0] + 0.000001*x[0]**3 + 2*x[1] + 0.000002/3*x[1]**3,
        "Constraints" : [

        ],
        "Optimal" : {
            "Solution" : [
                679.9453, 1026.067, 0.1188764, -0.3962336
            ],
            "Evaluation" : 5126.4981
        }
    },

    "G6" : {
        "Equation" : lambda x: (x[0] - 10)**3 + (x[1] - 20)**3,
        "Constraints" : [

        ],
        "Optimal" : {
            "Solution" : [
                14.095, 0.84296
            ],
            "Evaluation" : -6961.81381
        }
    }
}