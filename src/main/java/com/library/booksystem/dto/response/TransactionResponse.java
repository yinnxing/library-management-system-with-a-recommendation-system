package com.library.booksystem.dto.response;

import com.library.booksystem.model.Book;
import lombok.*;
import lombok.experimental.FieldDefaults;

import java.time.LocalDateTime;
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class TransactionResponse {
    private String transactionId;
    private BookResponse book;
    private LocalDateTime borrowDate;
    private LocalDateTime dueDate;
    private LocalDateTime returnDate;
    private String status;
}
