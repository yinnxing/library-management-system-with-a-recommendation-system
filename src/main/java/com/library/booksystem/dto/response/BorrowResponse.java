package com.library.booksystem.dto.response;

import lombok.*;
import lombok.experimental.FieldDefaults;

@Data
@NoArgsConstructor
@AllArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE)
@Builder
public class BorrowResponse {
    String userId;
    Integer bookId;
    String transactionId;
    private String expiryDate;
    private String status;
}
