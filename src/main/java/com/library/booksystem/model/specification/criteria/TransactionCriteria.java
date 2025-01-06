package com.library.booksystem.model.specification.criteria;

import com.library.booksystem.enums.TransactionStatus;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;
@Getter
@Setter
public final class TransactionCriteria {
    private String userId;
    private String bookId;
    private LocalDateTime borrowDateStart;
    private LocalDateTime borrowDateEnd;
    private LocalDateTime dueDateStart;
    private LocalDateTime dueDateEnd;
    private LocalDateTime returnDateStart;
    private LocalDateTime returnDateEnd;
    private TransactionStatus status;
    private String q;
}
