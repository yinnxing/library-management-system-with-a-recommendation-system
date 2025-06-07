package com.library.booksystem.model;

import com.library.booksystem.enums.TransactionStatus;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.math.BigDecimal;
import java.time.LocalDateTime;
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "Transactions")

public class Transaction {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private String transactionId;
    @ManyToOne
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @ManyToOne
    @JoinColumn(name = "book_id", nullable = false)
    private Book book;

    @Column(name = "borrow_date", updatable = false)
    private LocalDateTime borrowDate = LocalDateTime.now();

    private LocalDateTime dueDate;
    private LocalDateTime returnDate;
    
    // Pickup deadline for PENDING transactions (3 days from borrow_date)
    private LocalDateTime pickupDeadline;
    
    // Overdue fee in VND
    @Column(precision = 10, scale = 2)
    private BigDecimal overdueFee = BigDecimal.ZERO;
    
    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false)
    private TransactionStatus status = TransactionStatus.PENDING;


}
