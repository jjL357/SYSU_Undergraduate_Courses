(define (problem prob)
 (:domain puzzle)
 (:objects num_1 num_2 num_3 num_4 num_5 num_6 num_7 num_8 num_9 num_10 num_11 num_12 num_13 num_14 num_15 - num
           pos_1 pos_2 pos_3 pos_4 pos_5 pos_6 pos_7 pos_8 pos_9 pos_10 pos_11 pos_12 pos_13 pos_14 pos_15 pos_16 - loc
 )
 (:init (neighbor pos_1 pos_2)(neighbor pos_1 pos_5)
        (neighbor pos_2 pos_1)(neighbor pos_2 pos_3)(neighbor pos_2 pos_6)
        (neighbor pos_3 pos_2)(neighbor pos_3 pos_4)(neighbor pos_3 pos_7)
        (neighbor pos_4 pos_8)(neighbor pos_4 pos_3)
        (neighbor pos_5 pos_1)(neighbor pos_5 pos_6)(neighbor pos_5 pos_9)
        (neighbor pos_6 pos_5)(neighbor pos_6 pos_7)(neighbor pos_6 pos_2)(neighbor pos_6 pos_10)
        (neighbor pos_7 pos_6)(neighbor pos_7 pos_8)(neighbor pos_7 pos_3)(neighbor pos_7 pos_11)
        (neighbor pos_8 pos_7)(neighbor pos_8 pos_4)(neighbor pos_8 pos_12)
        (neighbor pos_9 pos_5)(neighbor pos_9 pos_10)(neighbor pos_9 pos_13)
        (neighbor pos_10 pos_9)(neighbor pos_10 pos_11)(neighbor pos_10 pos_14)(neighbor pos_10 pos_6)
        (neighbor pos_11 pos_10)(neighbor pos_11 pos_12)(neighbor pos_11 pos_7)(neighbor pos_11 pos_15)
        (neighbor pos_12 pos_11)(neighbor pos_12 pos_8)(neighbor pos_12 pos_16)
        (neighbor pos_13 pos_9)(neighbor pos_13 pos_14)
        (neighbor pos_14 pos_13)(neighbor pos_14 pos_15)(neighbor pos_14 pos_10)
        (neighbor pos_15 pos_14)(neighbor pos_15 pos_16)(neighbor pos_15 pos_11)
        (neighbor pos_16 pos_15)(neighbor pos_16 pos_12)
        
        ;(at num_14 pos_1)
        (at num_5 pos_2)
        (at num_15 pos_3)
        (at num_14 pos_4)
        (at num_7 pos_5)
        (at num_9 pos_6)
        (at num_6 pos_7)
        (at num_13 pos_8)
        (at num_1 pos_9)
        (at num_2 pos_10)
        (at num_12 pos_11)
        (at num_10 pos_12)
        (at num_8 pos_13)
        (at num_11 pos_14)
        (at num_4 pos_15)
        (at num_3 pos_16)
        (zero_at  pos_1)
;case 1
; 11 3 1 7
;4 6 8 2
;15 9 10 13
;14 12 5 0
;case2
;14 10 6 0
;4 9 1 8
;2 3 5 11
;12 13 7 15
;case3
;0 5 15 14
;7 9 6 13
;1 2 12 10
;8 11 4 3
;case4
;6 10 3 15
;14 8 7 11
;5 1 0 2
;13 12 9 4
  )
 (:goal (and(at num_1 pos_1)(at num_2 pos_2)(at num_3 pos_3)
        (at num_4 pos_4)
        (at num_5 pos_5)
        (at num_6 pos_6)
        (at num_7 pos_7)
        (at num_8 pos_8)
        (at num_9 pos_9)
        (at num_10 pos_10)
        (at num_11 pos_11)
        (at num_12 pos_12)
        (at num_13 pos_13)
        (at num_14 pos_14)
        (at num_15 pos_15)
        (zero_at  pos_16)))
)

