package com.library.booksystem.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;
import java.util.List;
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "books")
public class Book {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "book_id")
    private Integer bookId;

    @Column(nullable = false)
    private String title;

    private String author;
    private String publisher;
    @Column(name = "publication_year")
    private Integer publicationYear;

    @Column(nullable = false, unique = true)
    private String isbn;

    private String genre;
    private String descriptions;
    @Column(name = "cover_image_url")
    private String coverImageUrl;
    private Integer quantity;

    @Column(name = "available_quantity")
    private Integer availableQuantity;

    @Column(name = "preview_link")
    private String previewLink;

    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt = LocalDateTime.now();

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    @OneToMany(mappedBy = "book", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<Transaction> transactions;

    @OneToMany(mappedBy = "book", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<Review> reviews;

    @OneToMany(mappedBy = "book", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<Wishlist> wishlists;

}
