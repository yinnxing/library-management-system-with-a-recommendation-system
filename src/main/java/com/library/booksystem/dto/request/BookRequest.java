package com.library.booksystem.dto.request;

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.FieldDefaults;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE)
public class BookRequest {
    String title;
    String author;
    String publisher;
    Integer publicationYear;
    String isbn;
    String genre;
    String descriptions;
    String coverImageUrl;
    Integer quantity;
    Integer availableQuantity;
    LocalDateTime createdAt;
    String previewLink;

}
