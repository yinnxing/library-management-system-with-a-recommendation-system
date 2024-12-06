package com.library.booksystem.dto.response;

import lombok.*;
import lombok.experimental.FieldDefaults;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE)
public class BookResponse {
    private Integer bookId;
    private String title;
    private String author;
    private String publisher;
    private Integer publicationYear;
    private String isbn;
    private String genre;
    private String descriptions;
    private String coverImageUrl;
    private Integer quantity;
    private Integer availableQuantity;
    private LocalDateTime createdAt;
    private String previewLink;

}
