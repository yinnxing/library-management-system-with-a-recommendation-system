package com.library.booksystem.model.specification.criteria;

import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;
@Getter
@Setter

public final class BookCriteria {
    private String title;
    private String author;
    private String publisher;
    private Integer publicationYearStart;
    private Integer publicationYearEnd;
    private String isbn;
    private String genre;
    private Boolean hasCoverImage;
    private LocalDateTime createdAtStart;
    private LocalDateTime createdAtEnd;
    private String q; // Tìm kiếm tự do
}
