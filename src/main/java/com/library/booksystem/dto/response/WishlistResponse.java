package com.library.booksystem.dto.response;

import lombok.AccessLevel;
import lombok.Builder;
import lombok.Data;
import lombok.experimental.FieldDefaults;

import java.time.LocalDateTime;

@Builder
@Data
@FieldDefaults(level = AccessLevel.PRIVATE)
public class WishlistResponse {
    Integer wishlistId;
    String userId;
    Integer bookId;
    LocalDateTime addedAt = LocalDateTime.now();

}
