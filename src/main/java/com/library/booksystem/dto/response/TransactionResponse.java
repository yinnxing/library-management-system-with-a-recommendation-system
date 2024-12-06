package com.library.booksystem.dto.response;

import lombok.*;
import lombok.experimental.FieldDefaults;

import java.util.UUID;

@Data
@NoArgsConstructor
@AllArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE)
@Builder
public class TransactionResponse {
    String userId;
    Integer bookId;
    String transactionId;
    private String expiryDate;
    private String status;
}
